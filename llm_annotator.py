import anthropic
import time
from typing import List, Tuple, Optional
import random
from Levenshtein import distance
from api_key import default_model
from llm_cache import LLMCache


def _format_demonstrations(demonstrations: List[Tuple[str, str]]) -> str:
    """
    Format demonstrations for inclusion in the prompt.

    Args:
        demonstrations (List[Tuple[str, str]]): List of (text, label) pairs

    Returns:
        str: Formatted demonstrations string
    """
    formatted = "Here are some examples:\n\n"
    for text, label in demonstrations:
        formatted += f"Text: {text}\nLabel: {label}\n\n"
    return formatted


class LLMAnnotator:
    def __init__(
        self,
        api_key: str,
        instruction: str,
        task_description: str,
        label_names: List[str],
        examples: List[Tuple[str, str]],  # Required examples for few-shot learning
        model: str = default_model,
        max_tokens_to_sample: int = 50,
        temperature: float = 0.2,
        examples_per_prompt: int = 5,
        max_retries: int = 3,
        retry_delay: int = 2,
        use_cache: bool = True,
        cache_path: str = "llm_cache.db",
    ):
        """
        Initialize the LLM Annotator with Anthropic API.

        Args:
            api_key (str): Anthropic API key
            instruction (str): Instruction for the annotation task
            task_description (str): Description of the task
            label_names (List[str]): List of possible label names
            examples (List[Tuple[str, str]]): Required examples for few-shot learning
            model (str): Anthropic model to use
            max_tokens_to_sample (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            examples_per_prompt (int): Number of examples to include in each prompt
            max_retries (int): Maximum number of API call retries
            retry_delay (int): Delay between retries in seconds
            use_cache (bool): Whether to use caching for LLM requests
            cache_path (str): Path to the SQLite cache database
        """
        if not examples or len(examples) == 0:
            raise ValueError("Examples must be provided for few-shot learning")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.instruction = instruction
        self.task_description = task_description
        self.label_names = label_names + ["unknown"]  # Add 'unknown' as a valid label
        self.examples = examples
        self.model = model
        self.max_tokens_to_sample = max_tokens_to_sample
        self.temperature = temperature
        self.examples_per_prompt = min(
            examples_per_prompt, len(examples)
        )  # Ensure we don't ask for more examples than we have
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.messages = []
        self.system_prompt = ""

        # Initialize cache if enabled
        self.use_cache = use_cache
        self.cache = LLMCache(cache_path) if use_cache else None

        # Track cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize the thread with Anthropic immediately
        self._initialize_thread()

    def _initialize_thread(self):
        """Initialize a new thread with system prompt and examples"""
        # Clear previous messages
        self.messages = []

        # Create system prompt with clear, concise instructions
        self.system_prompt = f"You are a text classifier for labels: {', '.join(self.label_names)}. Respond with ONLY the label name, nothing else."

        # Add task description and format instructions
        task_message = f"{self.instruction}\n\n{self.task_description}\n\n"

        # Select a subset of examples if we have more than needed
        if len(self.examples) > self.examples_per_prompt:
            selected_demos = random.sample(self.examples, self.examples_per_prompt)
        else:
            selected_demos = self.examples

        # Add examples to the task message
        task_message += _format_demonstrations(selected_demos)

        # Add instructions about the expected response format
        task_message += "\nFor each text I send, respond with only the appropriate label from the list above. No explanations or additional text."

        # Add initial user message to our tracked messages
        self.messages.append({"role": "user", "content": task_message})

        print("\n=== Initializing Thread ===")
        print(f"System prompt: {self.system_prompt}")
        print(f"Task message: {task_message}")
        print("Initializing thread with Anthropic...")

        # Make the initial API call to establish the thread
        try:
            # Make the actual API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens_to_sample,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=self.messages,
            )

            # Add assistant's response to our message history
            if isinstance(response, dict):
                # This is a cached response
                content = response["content"][0]["text"]
            else:
                # This is a direct API response
                content = response.content[0].text

            self.messages.append({"role": "assistant", "content": content})
            print("Thread initialized successfully.")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize thread with Anthropic: {e}")

    def _call_anthropic_api(self, text: str) -> str | None:
        """
        Call the Anthropic API with retries, using the existing thread.

        Args:
            text (str): The text to classify

        Returns:
            str: The classified label
        """
        print(f"\n=== API Request ===")
        print(f"Text to classify: {text}")

        # Create a short, focused message for classification
        message = {"role": "user", "content": f"Text: {text}"}

        # Add to our tracked messages
        message_list = self.messages + [message]

        for attempt in range(self.max_retries):
            try:
                # Check cache first if enabled
                cached_response = None
                if self.use_cache:
                    cached_response = self.cache.get(self.model, text, message_list)

                if cached_response:
                    print("Using cached response")
                    response = cached_response
                    self.cache_hits += 1

                    # Get the raw response from cached content
                    if isinstance(response, dict) and "content" in response:
                        raw_response = response["content"][0]["text"].strip()
                    else:
                        # Fallback if cache structure changes
                        raw_response = str(response).strip()
                else:
                    # Send only the necessary context to the API
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens_to_sample,
                        temperature=self.temperature,
                        system=self.system_prompt,
                        messages=message_list,
                    )

                    # Store in cache if enabled
                    if self.use_cache:
                        self.cache.store(
                            self.model,
                            text,
                            message_list,
                            response.model_dump(),
                        )

                    # Get the raw response
                    raw_response = response.content[0].text.strip()
                    self.cache_misses += 1

                print("\n=== API Response ===")
                print(f"Raw response: {raw_response}")

                # Use Levenshtein distance to find the closest matching label
                best_match = self._find_best_label_match(raw_response)

                print(f"Best matching label: {best_match}")

                # Add the assistant's response to our tracked messages
                self.messages.append({"role": "assistant", "content": raw_response})

                return best_match

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(
                        f"API call failed with error: {e}. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to call Anthropic API after {self.max_retries} attempts: {e}"
                    )
        return None

    def _find_best_label_match(self, response: str) -> str:
        """
        Find the best matching label using Levenshtein distance.

        Args:
            response (str): The model's response text

        Returns:
            str: The best matching label
        """
        # Clean up the response
        response = response.lower().strip()

        # Simple case: exact match
        for label in self.label_names:
            if label.lower() == response:
                return label

        # Try to extract label if the response includes "label:" format
        if ":" in response:
            parts = response.split(":")
            potential_label = parts[-1].strip()
            for label in self.label_names:
                if label.lower() == potential_label:
                    return label

        # Use Levenshtein distance for fuzzy matching
        best_label = None
        best_distance = float("inf")

        for label in self.label_names:
            dist = distance(label.lower(), response)
            if dist < best_distance:
                best_distance = dist
                best_label = label

        # If the best distance is still too high, return "unknown"
        if best_distance > len(best_label) / 2:
            return "unknown"

        return best_label

    def annotate_batch(
        self,
        unlabeled_batch: List[str],
        demonstrations: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Annotate a batch of unlabeled examples using the initialized thread.

        Args:
            unlabeled_batch (List[str]): List of text examples to annotate
            demonstrations (Optional[List[Tuple[str, str]]]): Optional demonstrations to use

        Returns:
            List[Tuple[str, str]]: List of (text, predicted_label) pairs
        """
        annotations = []

        # If demonstrations are provided, update the thread
        if demonstrations is not None:
            # Save current messages
            original_messages = self.messages
            # Reinitialize with new demonstrations
            self.examples = demonstrations
            self._initialize_thread()

        # Process examples one by one, using the already initialized thread
        for text in unlabeled_batch:
            # Call API and get label
            label = self._call_anthropic_api(text)
            annotations.append((text, label))

        # Restore original messages if they were changed
        if demonstrations is not None:
            self.messages = original_messages

        # Print cache statistics
        if self.use_cache:
            print(f"\n=== Cache Statistics ===")
            print(f"Cache hits: {self.cache_hits}, Cache misses: {self.cache_misses}")
            print(
                f"Hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.2f}%"
            )

            # Reset counters for next batch
            self.cache_hits = 0
            self.cache_misses = 0

        return annotations

    def refresh_thread(self):
        """
        Refresh the thread with a new set of examples. Useful for long runs or to change examples.
        """
        self._initialize_thread()

    def get_cache_stats(self):
        """
        Get statistics about the cache usage.

        Returns:
            Dict: Statistics about the cache
        """
        if self.use_cache:
            return self.cache.get_stats()
        return {"enabled": False}
