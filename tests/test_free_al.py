# Unit and Integration Tests for FreeAL
import pytest
from unittest.mock import MagicMock, patch, mock_open

from free_al import FreeAL


@pytest.fixture
def mock_components():
    """Create mock components for testing FreeAL"""
    llm_annotator = MagicMock()
    slm_trainer = MagicMock()
    sample_selector = MagicMock()

    # Configure mock behaviors
    llm_annotator.annotate_batch.return_value = [
        ("text1", "positive"),
        ("text2", "negative"),
    ]
    slm_trainer.calculate_sample_losses.return_value = [
        ("text1", "positive", 0.1),
        ("text2", "negative", 0.2),
    ]
    slm_trainer.evaluate.return_value = 0.85
    sample_selector.filter_clean_samples.return_value = [
        ("text1", "positive"),
        ("text2", "negative"),
    ]
    sample_selector.select_demonstrations.return_value = [
        ("text1", "positive"),
        ("text2", "negative"),
    ]

    return {
        "llm_annotator": llm_annotator,
        "slm_trainer": slm_trainer,
        "sample_selector": sample_selector,
    }


@pytest.fixture
def freeal_instance(mock_components, tmp_path):
    """Create a FreeAL instance with mock components"""
    return FreeAL(
        llm_annotator=mock_components["llm_annotator"],
        slm_trainer=mock_components["slm_trainer"],
        sample_selector=mock_components["sample_selector"],
        label_names=["positive", "negative"],
        log_dir=str(tmp_path),
    )


def test_init(freeal_instance, tmp_path):
    """Test FreeAL initialization"""
    assert freeal_instance.current_round == 0
    assert freeal_instance.annotations == {}
    assert freeal_instance.demonstration_pool == []
    assert freeal_instance.label_names == ["positive", "negative"]
    assert freeal_instance.log_dir == str(tmp_path)
    assert tmp_path.exists()


def test_sample_initial_data(freeal_instance):
    """Test initial data sampling"""
    initial_data = [("example1", "positive"), ("example2", "negative")]
    result = freeal_instance._sample_initial_data(initial_data)

    assert result == initial_data
    assert freeal_instance.demonstration_pool == initial_data
    assert freeal_instance.annotations == {
        "example1": "positive",
        "example2": "negative",
    }


@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_log_state(mock_json_dump, mock_file, freeal_instance):
    """Test state logging functionality"""
    freeal_instance.current_round = 1
    freeal_instance.annotations = {"text1": "positive"}
    freeal_instance.demonstration_pool = [("text1", "positive")]

    freeal_instance._log_state({"val_accuracy": 0.85})

    assert mock_file.call_count == 3  # Three files should be opened
    assert mock_json_dump.call_count == 3  # Three JSON dumps should occur
    assert "val_accuracy" in freeal_instance.metrics


@patch("time.sleep")  # Mock sleep to speed up tests
def test_run_iteration(mock_sleep, freeal_instance, mock_components):
    """Test running a single iteration"""
    unlabeled_data = ["text1", "text2"]
    validation_data = [("val_text1", "positive"), ("val_text2", "negative")]

    # Run iteration
    with patch("tqdm.tqdm", lambda x: x):  # Mock tqdm to avoid progress bar in tests
        result = freeal_instance.run_iteration(unlabeled_data, validation_data)

    # Verify LLM annotator was called
    mock_components["llm_annotator"].annotate_batch.assert_called_once()

    # Verify SLM trainer was called
    mock_components["slm_trainer"].train.assert_called_once()
    mock_components["slm_trainer"].calculate_sample_losses.assert_called_once()
    mock_components["slm_trainer"].evaluate.assert_called_once()

    # Verify sample selector was called
    mock_components["sample_selector"].filter_clean_samples.assert_called_once()
    mock_components["sample_selector"].select_demonstrations.assert_called_once()

    # Check results
    assert result["round"] == 1
    assert "annotations" in result
    assert "demonstration_pool" in result
    assert "clean_samples" in result
    assert result["val_accuracy"] == 0.85

    # Check state updates
    assert freeal_instance.current_round == 1
    assert len(freeal_instance.annotations) > 0


def test_run_full_loop(freeal_instance, mock_components):
    """Test running the full FreeAL loop"""
    unlabeled_data = ["text1", "text2", "text3"]
    initial_labeled_data = [("example1", "positive"), ("example2", "negative")]
    validation_data = [("val_text1", "positive"), ("val_text2", "negative")]

    # Configure mock for convergence testing
    mock_components["slm_trainer"].evaluate.return_value = 0.90

    # Run full loop with patched methods to avoid actual file operations
    with (
        patch.object(freeal_instance, "_log_state"),
        patch.object(
            freeal_instance,
            "run_iteration",
            return_value={"val_accuracy": 0.90, "clean_samples": 2},
        ),
        patch("os.makedirs"),
    ):
        result = freeal_instance.run_full_loop(
            unlabeled_data=unlabeled_data,
            initial_labeled_data=initial_labeled_data,
            validation_data=validation_data,
            iterations=2,
        )

    # Check results
    assert "rounds_completed" in result
    assert "final_annotations" in result
    assert "final_demonstration_pool" in result
    assert "final_accuracy" in result
    assert "model_path" in result


def test_predict(freeal_instance, mock_components):
    """Test prediction functionality"""
    texts = ["This is great!", "This is terrible."]
    mock_components["slm_trainer"].predict.return_value = ["positive", "negative"]

    predictions = freeal_instance.predict(texts)

    mock_components["slm_trainer"].predict.assert_called_once_with(texts)
    assert predictions == ["positive", "negative"]


def test_convergence_early_stopping(freeal_instance):
    """Test early stopping based on convergence threshold"""
    unlabeled_data = ["text1", "text2", "text3"]
    initial_labeled_data = [("example1", "positive"), ("example2", "negative")]
    validation_data = [("val_text1", "positive"), ("val_text2", "negative")]

    # First iteration has significant improvement
    first_result = {"val_accuracy": 0.80, "clean_samples": 2}
    # Second iteration has minimal improvement (below threshold)
    second_result = {"val_accuracy": 0.803, "clean_samples": 2}

    # Mock run_iteration to return our predefined results
    with (
        patch.object(freeal_instance, "_log_state"),
        patch.object(freeal_instance, "_sample_initial_data"),
        patch.object(
            freeal_instance, "run_iteration", side_effect=[first_result, second_result]
        ),
        patch("os.makedirs"),
    ):

        result = freeal_instance.run_full_loop(
            unlabeled_data=unlabeled_data,
            initial_labeled_data=initial_labeled_data,
            validation_data=validation_data,
            iterations=3,  # Set to 3, but should stop after 2
            convergence_threshold=0.005,
        )

    # Should have called run_iteration only twice due to early stopping
    assert freeal_instance.run_iteration.call_count == 2


# Integration test with more realistic data flow
def test_integration_data_flow():
    """Test the data flow through the FreeAL system with more realistic mocks"""

    # Create more sophisticated mocks that maintain state
    class MockLLMAnnotator:
        def annotate_batch(self, texts, demonstrations):
            return [
                (text, "positive" if "good" in text.lower() else "negative")
                for text in texts
            ]

    class MockSLMTrainer:
        def train(self, labeled_data, validation_data, epochs):
            self.trained_data = labeled_data
            return {"loss": 0.5}

        def calculate_sample_losses(self, samples):
            return [
                (text, label, 0.1 if "good" in text.lower() else 0.8)
                for text, label in samples
            ]

        def evaluate(self, validation_data):
            return 0.85

        def predict(self, texts):
            return [
                "positive" if "good" in text.lower() else "negative" for text in texts
            ]

    class MockSampleSelector:
        def filter_clean_samples(self, samples_with_loss):
            # Keep only samples with low loss
            return [
                (text, label) for text, label, loss in samples_with_loss if loss < 0.5
            ]

        def select_demonstrations(self, samples, num_per_class):
            # Group by class
            pos_samples = [s for s in samples if s[1] == "positive"]
            neg_samples = [s for s in samples if s[1] == "negative"]

            # Select up to num_per_class from each
            pos_selected = pos_samples[:num_per_class]
            neg_selected = neg_samples[:num_per_class]

            return pos_selected + neg_selected

    # Create FreeAL instance with our stateful mocks
    with (
        patch("os.makedirs"),
        patch("builtins.open", new_callable=mock_open),
        patch("json.dump"),
    ):
        freeal = FreeAL(
            llm_annotator=MockLLMAnnotator(),
            slm_trainer=MockSLMTrainer(),
            sample_selector=MockSampleSelector(),
            label_names=["positive", "negative"],
            log_dir="test_logs",
        )

        # Test data
        unlabeled_data = [
            "This product is good and I like it",
            "This is terrible and I hate it",
            "Good experience overall",
            "Bad customer service",
        ]

        initial_data = [
            ("I had a good time", "positive"),
            ("This was awful", "negative"),
        ]

        # Run a single iteration
        with patch("tqdm.tqdm", lambda x: x), patch("time.sleep"):
            freeal._sample_initial_data(initial_data)
            result = freeal.run_iteration(unlabeled_data)

        # Verify data flow
        assert len(freeal.annotations) == 6  # 2 initial + 4 new
        assert len(freeal.demonstration_pool) > 0
        assert result["clean_samples"] > 0
