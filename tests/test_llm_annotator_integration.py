import unittest
import os
from llm_annotator import LLMAnnotator
from api_key import anthropic_api_key


class TestLLMAnnotatorIntegration(unittest.TestCase):
    """Integration tests for LLMAnnotator.

    These tests require an actual Anthropic API key and make real API calls.
    They should be run sparingly to avoid unnecessary API costs.

    To run these tests:
        ANTHROPIC_API_KEY=your_key pytest tests/test_integration.py -v
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Skip all tests if no API key is provided
        cls.api_key = os.environ.get("ANTHROPIC_API_KEY", anthropic_api_key)
        if not cls.api_key:
            raise unittest.SkipTest(
                "Skipping integration tests: No ANTHROPIC_API_KEY environment variable found"
            )

        # Define examples that will be used across tests
        cls.examples = [
            ("I love this product!", "positive"),
            ("This is the worst purchase ever.", "negative"),
            ("It works as expected.", "neutral"),
        ]

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.instruction = "Classify the sentiment of the following text."
        self.task_description = "Determine if the text expresses a positive, negative, or neutral sentiment."
        self.label_names = ["positive", "negative", "neutral"]

        # Create the annotator with actual API key and required examples
        self.annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
        )

    def test_single_annotation(self):
        """Test annotation of a single example with obvious sentiment."""
        # Test positive example
        result = self.annotator._call_anthropic_api(
            "I absolutely love this product! It's amazing!"
        )
        self.assertEqual(result, "positive")

        # Test negative example
        result = self.annotator._call_anthropic_api(
            "This is terrible, I want my money back."
        )
        self.assertEqual(result, "negative")

        # Test neutral example
        result = self.annotator._call_anthropic_api("The product arrived today.")
        self.assertEqual(result, "neutral")

    def test_batch_annotation(self):
        """Test batch annotation of multiple examples."""
        # Create a new annotator instance to ensure a fresh thread
        annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
        )

        batch = [
            "This exceeded all my expectations!",
            "I'm very disappointed with this purchase.",
            "The item was delivered on time.",
        ]

        results = annotator.annotate_batch(batch)

        # Check that we got results for all examples
        self.assertEqual(len(results), len(batch))

        # Check that the results make sense
        self.assertEqual(results[0][1], "positive")
        self.assertEqual(results[1][1], "negative")
        self.assertEqual(results[2][1], "neutral")

    def test_with_different_examples(self):
        """Test annotation with a different set of examples."""
        # Create different examples
        different_examples = [
            ("The customer service was exceptional!", "positive"),
            ("I regret buying this product.", "negative"),
            ("The delivery was neither fast nor slow.", "neutral"),
        ]

        # Create a new annotator with different examples
        different_annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=different_examples,
        )

        # Test annotation
        result = different_annotator._call_anthropic_api(
            "I'm really happy with this purchase."
        )
        self.assertEqual(result, "positive")

    def test_refresh_thread(self):
        """Test that refreshing the thread works correctly."""
        # Get results before refreshing
        result_before = self.annotator._call_anthropic_api(
            "This is a fantastic product!"
        )
        self.assertEqual(result_before, "positive")

        # Refresh the thread
        self.annotator.refresh_thread()

        # Test after refreshing
        result_after = self.annotator._call_anthropic_api(
            "This is a fantastic product!"
        )
        self.assertEqual(result_after, "positive")

    def test_with_edge_cases(self):
        """Test annotation with edge cases and ambiguous text."""
        edge_cases = [
            "The product is okay.",  # Ambiguous, could be neutral
            "It's not great, but not terrible either.",  # Mixed sentiment
            "!!!!!",  # Minimal content
            "I don't know what to think about this product yet.",  # Uncertainty
        ]

        for text in edge_cases:
            result = self.annotator._call_anthropic_api(text)
            # Just verify we get a valid label (not checking specific values)
            self.assertIn(result, self.label_names + ["unknown"])

    def test_custom_examples_per_prompt(self):
        """Test with custom number of examples per prompt."""
        # Use all examples (no sampling)
        all_examples_annotator = LLMAnnotator(
            api_key=self.api_key,
            instruction=self.instruction,
            task_description=self.task_description,
            label_names=self.label_names,
            examples=self.examples,
            examples_per_prompt=len(self.examples),  # Use all examples
        )

        result = all_examples_annotator._call_anthropic_api(
            "This product changed my life for the better!"
        )
        self.assertEqual(result, "positive")

        # Use just one example (minimal context)
        if len(self.examples) > 1:
            minimal_context_annotator = LLMAnnotator(
                api_key=self.api_key,
                instruction=self.instruction,
                task_description=self.task_description,
                label_names=self.label_names,
                examples=self.examples,
                examples_per_prompt=1,  # Use just one example
            )

            result = minimal_context_annotator._call_anthropic_api(
                "This product changed my life for the better!"
            )
            self.assertEqual(result, "positive")


if __name__ == "__main__":
    unittest.main()
