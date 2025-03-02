import unittest
from unittest.mock import MagicMock, patch, ANY
from collections import namedtuple
import os
import tempfile
import json

# Import the LLMAnnotator class
# Assuming the modified code is in a file named llm_annotator.py
from llm_annotator import LLMAnnotator

# Create mock types to simulate Anthropic's response structure
MockContent = namedtuple('MockContent', ['text'])
MockResponse = namedtuple('MockResponse', ['content', 'model_dump'])


class TestLLMAnnotator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.instruction = "Classify the sentiment of the text"
        self.task_description = "Determine if the text expresses positive, negative, or neutral sentiment"
        self.label_names = ["positive", "negative", "neutral"]
        self.examples = [
            ("I love this product!", "positive"),
            ("This is terrible, don't buy it", "negative"),
            ("The product arrived on time", "neutral")
        ]

        # Create a temporary file for the cache
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.cache_path = self.temp_db.name

        # Create mock for Anthropic client
        self.mock_anthropic_patcher = patch('anthropic.Anthropic', autospec=True)
        self.mock_anthropic = self.mock_anthropic_patcher.start()

        # Create a mock instance for the Anthropic client
        self.mock_client = MagicMock()
        self.mock_anthropic.return_value = self.mock_client

        # Set up mock response for messages.create
        mock_content = MockContent(text="positive")
        mock_response = MockResponse(
            content=[mock_content],
            model_dump=lambda: {"content": [{"text": "positive"}]}
        )
        self.mock_client.messages.create.return_value = mock_response

        # Mock for LLMCache
        self.mock_cache_patcher = patch('llm_annotator.LLMCache')
        self.mock_cache_class = self.mock_cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache_class.return_value = self.mock_cache

        # By default, cache misses all requests
        self.mock_cache.get.return_value = None

    def tearDown(self):
        """Tear down test fixtures"""
        self.mock_anthropic_patcher.stop()
        self.mock_cache_patcher.stop()

        # Remove temporary cache file
        if os.path.exists(self.cache_path):
            os.unlink(self.cache_path)

    def test_init_requires_examples(self):
        """Test that initialization requires examples for few-shot learning"""
        with self.assertRaises(ValueError):
            LLMAnnotator(
                api_key=self.api_key,
                instruction=self.instruction,
                task_description=self.task_description,
                label_names=self.label_names,
                examples=[]  # Empty list should raise ValueError
            )

    def test_init_creates_anthropic_client(self):
        """Test that initialization creates an Anthropic client"""
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=True,
            cache_path=self.cache_path
        )

        # Check if Anthropic client was created with correct API key
        self.mock_anthropic.assert_called_once_with(api_key=self.api_key)

        # Check if initial thread was created
        self.mock_client.messages.create.assert_called_once()

        # Check if cache was initialized
        self.mock_cache_class.assert_called_once_with(self.cache_path)

    def test_annotate_batch_calls_api_correctly(self):
        """Test that annotate_batch calls the Anthropic API correctly for each text"""
        # Create annotator with mock
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=True,
            cache_path=self.cache_path
        )

        # Reset mock to clear the initialization call
        self.mock_client.messages.create.reset_mock()
        self.mock_cache.get.reset_mock()
        self.mock_cache.store.reset_mock()

        # Set up different responses for each API call
        responses = [
            MockResponse(
                content=[MockContent(text="positive")],
                model_dump=lambda: {"content": [{"text": "positive"}]}
            ),
            MockResponse(
                content=[MockContent(text="negative")],
                model_dump=lambda: {"content": [{"text": "negative"}]}
            ),
            MockResponse(
                content=[MockContent(text="neutral")],
                model_dump=lambda: {"content": [{"text": "neutral"}]}
            )
        ]
        self.mock_client.messages.create.side_effect = responses

        # Call annotate_batch with test data
        texts = [
            "I really enjoyed this movie!",
            "The service was awful.",
            "It was okay, nothing special."
        ]
        results = annotator.annotate_batch(texts)

        # Check API was called the correct number of times (once per text)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Check cache was queried for each text
        self.assertEqual(self.mock_cache.get.call_count, 3)

        # Check cache was updated for each API call
        self.assertEqual(self.mock_cache.store.call_count, 3)

        # Check results
        expected_results = [
            ("I really enjoyed this movie!", "positive"),
            ("The service was awful.", "negative"),
            ("It was okay, nothing special.", "neutral")
        ]
        self.assertEqual(results, expected_results)

    def test_find_best_label_match(self):
        """Test the _find_best_label_match method with various inputs"""
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=False
        )

        # Test exact match
        self.assertEqual(annotator._find_best_label_match("positive"), "positive")

        # Test case insensitivity
        self.assertEqual(annotator._find_best_label_match("POSITIVE"), "positive")

        # Test with label: format
        self.assertEqual(annotator._find_best_label_match("Label: negative"), "negative")

        # Test fuzzy matching
        self.assertEqual(annotator._find_best_label_match("positiv"), "positive")

        # Test completely unrelated text returns unknown
        self.assertEqual(annotator._find_best_label_match("xyzabc"), "unknown")

    def test_refresh_thread(self):
        """Test that refresh_thread reinitializes the thread"""
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=True,
            cache_path=self.cache_path
        )

        # Reset mock to clear the initialization call
        self.mock_client.messages.create.reset_mock()
        self.mock_cache.get.reset_mock()
        self.mock_cache.store.reset_mock()

        # Call refresh_thread
        annotator.refresh_thread()

        # Check if a new thread was created
        self.mock_client.messages.create.assert_called_once()

        # Check if cache was queried
        self.mock_cache.get.assert_called_once()

        # Check if cache was updated
        self.mock_cache.store.assert_called_once()

    @patch('random.sample')
    def test_examples_sampling(self, mock_random_sample):
        """Test that examples are sampled correctly when there are more than examples_per_prompt"""
        # Set up the mock to return a predetermined sample
        sample = [
            ("I love this product!", "positive"),
            ("This is terrible, don't buy it", "negative")
        ]
        mock_random_sample.return_value = sample

        # Create an annotator with more examples than examples_per_prompt
        extended_examples = self.examples + [
            ("Amazing experience!", "positive"),
            ("I hate it", "negative"),
            ("Just as expected", "neutral")
        ]

        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=extended_examples,
            examples_per_prompt=2,  # We want to sample only 2 examples
            use_cache=True,
            cache_path=self.cache_path
        )

        # Check that random.sample was called with the right arguments
        mock_random_sample.assert_called_once_with(extended_examples, 2)

    def test_api_retry_mechanism(self):
        """Test that the API retry mechanism works correctly"""
        # Create an annotator
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            max_retries=3,
            retry_delay=0,  # Set to 0 for faster tests
            use_cache=True,
            cache_path=self.cache_path
        )

        # Reset mock to clear the initialization call
        self.mock_client.messages.create.reset_mock()
        self.mock_cache.get.reset_mock()
        self.mock_cache.store.reset_mock()

        # Set up the mock to fail twice then succeed
        self.mock_client.messages.create.side_effect = [
            Exception("API Error"),
            Exception("API Error"),
            MockResponse(
                content=[MockContent(text="positive")],
                model_dump=lambda: {"content": [{"text": "positive"}]}
            )
        ]

        # Call the method that uses the API
        result = annotator._call_anthropic_api("Test text")

        # Check that API was called 3 times (2 failures, 1 success)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Check that cache was queried 3 times (once for each attempt)
        self.assertEqual(self.mock_cache.get.call_count, 3)

        # Check that cache was updated once (on success)
        self.assertEqual(self.mock_cache.store.call_count, 1)

        # Check the final result
        self.assertEqual(result, "positive")

    def test_cache_hit(self):
        """Test that cache hits prevent API calls"""
        # Create an annotator
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=True,
            cache_path=self.cache_path
        )

        # Reset mocks
        self.mock_client.messages.create.reset_mock()
        self.mock_cache.get.reset_mock()
        self.mock_cache.store.reset_mock()

        # Manually reset the cache counters since they persist across the object
        annotator.cache_hits = 0
        annotator.cache_misses = 0

        # Set up cache to return a hit
        self.mock_cache.get.return_value = {
            "content": [{"text": "positive"}]
        }

        # Call the API method
        result = annotator._call_anthropic_api("Test text")

        # Check that API was not called
        self.mock_client.messages.create.assert_not_called()

        # Check that cache was queried
        self.mock_cache.get.assert_called_once()

        # Check that cache was not updated
        self.mock_cache.store.assert_not_called()

        # Check the result came from cache
        self.assertEqual(result, "positive")

        # Check cache hit counter was incremented
        self.assertEqual(annotator.cache_hits, 1)
        self.assertEqual(annotator.cache_misses, 0)

    def test_cache_disabled(self):
        """Test that everything works when cache is disabled"""
        # Create an annotator with caching disabled
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=False
        )

        # Check that cache was not created
        self.mock_cache_class.assert_not_called()

        # Reset API mock
        self.mock_client.messages.create.reset_mock()

        # Call the API method
        result = annotator._call_anthropic_api("Test text")

        # Check that API was called
        self.mock_client.messages.create.assert_called_once()

        # Check the result
        self.assertEqual(result, "positive")

    def test_get_cache_stats(self):
        """Test the get_cache_stats method"""
        # Create an annotator with caching enabled
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=True,
            cache_path=self.cache_path
        )

        # Set up mock to return stats
        self.mock_cache.get_stats.return_value = {
            "total_entries": 10,
            "by_model": {"claude-3": 10}
        }

        # Call the method
        stats = annotator.get_cache_stats()

        # Check results
        self.assertEqual(stats["total_entries"], 10)
        self.assertEqual(stats["by_model"]["claude-3"], 10)

        # Create an annotator with caching disabled
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=False
        )

        # Call the method
        stats = annotator.get_cache_stats()

        # Check results
        self.assertEqual(stats, {"enabled": False})

    def test_annotate_batch_with_demonstrations(self):
        """Test that annotate_batch uses provided demonstrations"""
        # Create annotator
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            use_cache=True,
            cache_path=self.cache_path
        )

        # Reset mocks
        self.mock_client.messages.create.reset_mock()

        # New demonstrations to use
        new_demos = [
            ("This is great!", "positive"),
            ("This is awful!", "negative")
        ]

        # Call annotate_batch with custom demonstrations
        texts = ["Test text"]
        annotator.annotate_batch(texts, demonstrations=new_demos)

        # Check that initialization was called with new demos
        self.assertEqual(self.mock_client.messages.create.call_count, 2)  # Init + annotation


if __name__ == "__main__":
    unittest.main()