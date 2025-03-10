import anthropic
import re
import sqlite3
import hashlib
import json
import random
from typing import List, Tuple, Dict, Any, Optional

from anthropic.types.messages import MessageBatch

from api_key import anthropic_api_key, default_model
from models.classification_result import ClassificationResult

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=anthropic_api_key)


@dataclass
class LLMResult:
    """Data class to store the question and its response."""
    question: str
    response: str


class ResponseCache:
    """Cache for LLM responses using SQLite."""

    def __init__(self, db_path="llm_response_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS responses
                       (
                           hash_key
                           TEXT
                           PRIMARY
                           KEY,
                           question
                           TEXT,
                           response
                           TEXT,
                           model
                           TEXT,
                           timestamp
                           INTEGER
                       )
                       ''')
        self.conn.commit()

    def get_hash_key(self, question: str, model: str) -> str:
        """Create a hash key from the question and model."""
        combined = f"{question}:{model}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, question: str, model: str) -> Optional[str]:
        """Get a cached response if it exists."""
        hash_key = self.get_hash_key(question, model)
        cursor = self.conn.cursor()
        cursor.execute("SELECT response FROM responses WHERE hash_key = ?", (hash_key,))
        result = cursor.fetchone()
        return result[0] if result else None

    def set(self, question: str, response: str, model: str):
        """Cache a response."""
        hash_key = self.get_hash_key(question, model)
        timestamp = int(time.time())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO responses (hash_key, question, response, model, timestamp) VALUES (?, ?, ?, ?, ?)",
            (hash_key, question, response, model, timestamp)
        )
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()


# Initialize cache
response_cache = ResponseCache()


def classify_texts_batch(
        system_prompt: str,
        detailed_task_description: str,
        texts: List[str],
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        batch_size: int = 100,
        poll_interval: int = 10
) -> List[LLMResult]:
    """
    Classify texts using Anthropic's Claude API with batch processing.
    Uses caching to avoid re-processing texts that were already analyzed.

    Args:
        system_prompt: The system prompt that defines the classification task
        detailed_task_description: The text content to be analyzed
        texts: List of questions/classification tasks to perform on the text
        model: Anthropic model to use (default: claude-3-haiku-20240307)
        max_tokens: Maximum tokens for response (default: 1024)
        batch_size: Maximum batch size (default: 50)
        poll_interval: Seconds between status checks (default: 10)

    Returns:
        List of ClassificationResult objects containing questions and their responses
    """
    results = []
    question_map = {}  # Maps custom_ids to questions

    # Check cache first for all questions
    uncached_questions = []
    uncached_indices = []

    print(f"Starting batch classification with model: {model}")
    print(f"Total questions to process: {len(texts)}")
    print("Checking cache for existing responses...")

    for i, question in enumerate(texts):
        cached_response = response_cache.get(question, model)
        if cached_response:
            print(f"Cache hit for question: {question[:30]}...")
            results.append(LLMResult(question=question, response=cached_response))
        else:
            uncached_questions.append(question)
            uncached_indices.append(i)

    print(f"Found {len(texts) - len(uncached_questions)} cached responses")
    print(f"Need to process {len(uncached_questions)} uncached questions")

    if not uncached_questions:
        print("All questions were found in cache, returning results")
        return results

    # Process uncached questions in batches
    for i in range(0, len(uncached_questions), batch_size):
        batch_questions = uncached_questions[i:i + batch_size]
        requests = []

        print(f"\nPreparing batch {i // batch_size + 1} of {(len(uncached_questions) + batch_size - 1) // batch_size}")
        print(f"Questions in this batch: {len(batch_questions)}")

        # Prepare requests for this batch
        for idx, question in enumerate(batch_questions):
            # Use the question index as the custom_id
            custom_id = f"q{i + idx}"
            question_map[custom_id] = question

            requests.append(
                Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=max_tokens,
                        system=[
                            {
                                "type": "text",
                                "text": system_prompt
                            },
                            {
                                "type": "text",
                                "text": detailed_task_description,
                                "cache_control": {"type": "ephemeral"}
                            }
                        ],
                        messages=[{
                            "role": "user",
                            "content": f"Example: {question}"
                        }]
                    )
                )
            )

        # Submit the batch
        try:
            print(f"Submitting batch to API...")
            start_time = time.time()
            message_batch = client.messages.batches.create(requests=requests)
            print(f"Batch submitted successfully. Batch ID: {message_batch.id}")
            print(f"Initial status: {message_batch.processing_status}")
            print(f"Request counts: {message_batch.request_counts}")

        except Exception as e:
            print(f"Batch submitted failed: {e}")
            return results  # Return whatever we have cached so far

        # Poll for completion
        print(f"Polling for batch completion...");
        failures = 0
        while True:
            try:
                message_batch = client.messages.batches.retrieve(message_batch.id)
            except Exception as e:
                print(f"Batch completion failed: {e}")
                failures += 1
                if failures > 3:
                    print(f"Batch completion failed. Not Retrying...")
                    return results  # Return whatever we have cached so far

            print(f"Status: {message_batch.processing_status}")
            print(f"Request counts: {message_batch.request_counts}")

            if message_batch.processing_status == "ended":
                print(f"Batch processing completed after {time.time() - start_time:.2f} seconds")
                break

            print(f"Waiting {poll_interval} seconds before checking again...")
            time.sleep(poll_interval)

        # Process results
        print(f"Retrieving batch results...")
        batch_responses = {}
        try:
            async_results = client.messages.batches.results(message_batch.id)
        except Exception as e:
            print(f"Batch results failed: {e}")
            return results  # Return whatever we have cached so far

        for r in async_results:
            rid = r.custom_id if hasattr(r, "custom_id") else r.id
            result = r.result
            succeeded = result.type == 'succeeded'
            if succeeded:
                txt = result.message.content[0].text

                if rid in question_map:
                    question = question_map[rid]
                    # Cache the response
                    response_cache.set(question, txt, model)
                    results.append(LLMResult(
                        question=question,
                        response=txt,
                    ))
                    print(f'success! Question: {question[:20]}, Response: {txt}')
                else:
                    print("No question found for id " + rid)

    # Sort results to maintain original order
    sorted_results = []
    result_dict = {r.question: r for r in results}

    for question in texts:
        if question in result_dict:
            sorted_results.append(result_dict[question])

    print(f"\nClassification complete. Processed {len(sorted_results)} questions.")
    return sorted_results


def batch_classify(
        texts: List[str],
        examples: List[Tuple[str, str]],
        label_names: List[str],
        task_description: str,
        model: str = default_model,
        temperature: float = 0.2,
) -> List[ClassificationResult]:
    """
    Classify multiple texts using Anthropic's Claude.
    Uses a SQLite cache to store and retrieve results for efficiency.
    Processes each text individually to ensure accuracy.

    Args:
        texts (List[str]): List of texts to classify
        examples (List[Tuple[str, str]]): List of (text, label) example pairs
        label_names (List[str]): List of possible label names
        task_description (str): Description of the classification task
        model (str): Anthropic model to use
        temperature (float): Sampling temperature

    Returns:
        List[ClassificationResult]: List of classification results
    """
    if not texts:
        return []

    # Format examples
    formatted_examples = "Here are some examples:\n\n"
    for i, (text, label) in enumerate(examples):
        formatted_examples += f"Example {i + 1}:\nText: \"{text}\"\nCorrect Label: \"{label}\"\n\n"

    # Create system prompt
    system_prompt = f"You are a text classifier for labels: {', '.join(label_names)}. When I send you a text to classify, respond only with the label and confidence score separated by a comma. For example: \"{label_names[0]},0.95\". Don't include any other text in your response."

    # Build the initial user message with task description and examples
    user_message = f"""

{task_description}

Here are some examples:

{formatted_examples}

I will send you texts to classify. For each text, respond with only the label and a confidence score (between 0 and 1) separated by a comma. No other text or explanation
For example, if you think a text is {label_names[0]} with 95% confidence, respond with exactly: {label_names[0]},0.95"""

    responses = classify_texts_batch(
        system_prompt=system_prompt,
        detailed_task_description=user_message,
        texts=texts,
        model=model)

    results = []
    for resp in responses:
        raw_label, confidence = parse_classification_response(resp.response, label_names)
        print(f"Classification result: {resp.question[:30]} ({raw_label}, {confidence})")
        result = ClassificationResult(
            text=resp.question,
            raw_label=raw_label,
            confidence=confidence,
            valid_labels=label_names,
        )
        results.append(result)
    return results


def parse_classification_response(response_text: str, valid_labels: List[str]) -> Tuple[str, float]:
    """
    Parse a classification response in the format "label,confidence"

    Args:
        response_text (str): Response text to parse
        valid_labels (List[str]): List of valid labels for validation

    Returns:
        Tuple[str, float]: Extracted label and confidence
    """
    # First, try to parse as CSV (label,confidence)
    parts = response_text.strip().split(',', 1)

    if len(parts) == 2:
        raw_label = parts[0].strip()

        # Try to extract confidence
        try:
            confidence = float(parts[1].strip())
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            # If we can't parse the confidence, try to find a number in the string
            confidence_match = re.search(r'(\d+\.\d+|\d+)', parts[1])
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            else:
                confidence = 0.5
    else:
        # If not in expected format, try to extract the label directly
        # First check if the response exactly matches any valid label
        if response_text.strip() in valid_labels:
            raw_label = response_text.strip()
            confidence = 0.9  # Assume high confidence for exact match
        else:
            # Otherwise try to find the first valid label in the response
            for label in valid_labels:
                if label.lower() in response_text.lower():
                    raw_label = label
                    break
            else:
                raw_label = response_text.strip()

            # Look for confidence value separately
            confidence_match = re.search(r'(\d+\.\d+|\d+)', response_text)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            else:
                confidence = 0.5

    return raw_label, confidence