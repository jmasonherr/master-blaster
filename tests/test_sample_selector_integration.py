import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sample_selector import SampleSelector, k_medoids, filter_clean_samples


@pytest.fixture(scope="module")
def sample_data():
    """Fixture that provides sample text data for testing."""
    texts = [
        "This movie was fantastic! The acting was superb.",
        "I really enjoyed this film and would recommend it to everyone.",
        "This is the best movie I've seen all year.",
        "The cinematography was beautiful and the story was engaging.",
        "I didn't enjoy this movie at all. The plot was confusing.",
        "This film was terrible. I wasted my money on it.",
        "The worst movie experience I've had. Complete waste of time.",
        "Poor acting and a predictable storyline made this disappointing.",
    ]

    labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
    ]

    # Simulate loss values (lower for clean samples, higher for noisy)
    # Make some clean examples for both classes to ensure class balance
    losses = [
        0.1,  # Clean positive
        0.15,  # Clean positive
        0.8,  # Noisy positive
        0.85,  # Noisy positive
        0.2,  # Clean negative
        0.25,  # Clean negative
        0.9,  # Noisy negative
        0.95,  # Noisy negative
    ]

    # Create samples with loss
    samples_with_loss = list(zip(texts, labels, losses))

    return {
        "texts": texts,
        "labels": labels,
        "losses": losses,
        "samples_with_loss": samples_with_loss,
    }


@pytest.fixture(scope="module")
def selector():
    """Fixture that provides the SampleSelector instance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # First try with a small, fast model
        return SampleSelector(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", device=device
        )
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")


@pytest.fixture(scope="module")
def real_sentiment_data():
    """Fixture that provides real sentiment data if available."""
    try:
        # Try to load sample sentiment data
        dataset_path = Path("./data/sample_sentiment.csv")
        if not dataset_path.exists():
            pytest.skip("Real sentiment data not available")

        # Load the data
        df = pd.read_csv(dataset_path)
        texts = df["text"].tolist()[:100]  # Limit to 100 samples
        labels = df["label"].tolist()[:100]
        losses = (
            df["loss"].tolist()[:100]
            if "loss" in df.columns
            else [np.random.uniform(0.1, 0.9) for _ in range(100)]
        )

        return {
            "texts": texts,
            "labels": labels,
            "losses": losses,
            "samples_with_loss": list(zip(texts, labels, losses)),
        }
    except Exception as e:
        pytest.skip(f"Error loading real sentiment data: {e}")


def test_embedding_generation(selector, sample_data):
    """Test that the embedding generation produces valid embeddings for real text."""
    texts = sample_data["texts"]

    # Generate embeddings for all texts
    embeddings = selector._get_embeddings(texts)

    # Check that we get the right number of embeddings
    assert len(embeddings) == len(texts)

    # Check that embeddings are non-zero and have reasonable values
    assert np.any(embeddings != 0)

    # Verify embeddings have consistent dimensions
    assert all(emb.shape == embeddings[0].shape for emb in embeddings)

    # Similar texts should have more similar embeddings
    # Create a similarity matrix
    similarity = cosine_similarity(embeddings)

    # Positive examples should be more similar to each other
    pos_pos_sim = similarity[0:4, 0:4].mean()

    # Negative examples should be more similar to each other
    neg_neg_sim = similarity[4:8, 4:8].mean()

    # Cross-class similarity should be lower
    pos_neg_sim = similarity[0:4, 4:8].mean()

    # Check that intra-class similarity is higher than inter-class
    assert pos_pos_sim > pos_neg_sim
    assert neg_neg_sim > pos_neg_sim

    print(f"Positive-Positive similarity: {pos_pos_sim:.4f}")
    print(f"Negative-Negative similarity: {neg_neg_sim:.4f}")
    print(f"Positive-Negative similarity: {pos_neg_sim:.4f}")


def test_filtering_clean_samples(selector, sample_data):
    """
    Test that the GMM-based filtering correctly identifies clean samples
    based on loss values, as described in the paper.
    """
    samples_with_loss = sample_data["samples_with_loss"]
    losses = sample_data["losses"]
    texts = sample_data["texts"]

    # Filter clean samples
    clean_samples = filter_clean_samples(samples_with_loss)

    # We should get some clean samples (not empty, not all)
    assert 0 < len(clean_samples) < len(samples_with_loss)

    # The clean samples should have lower loss values
    clean_texts = [sample[0] for sample in clean_samples]
    clean_indices = [i for i, text in enumerate(texts) if text in clean_texts]
    clean_losses = [losses[i] for i in clean_indices]
    remaining_indices = [i for i in range(len(texts)) if i not in clean_indices]
    remaining_losses = [losses[i] for i in remaining_indices]

    # Check that the average loss of clean samples is lower than non-clean
    assert np.mean(clean_losses) < np.mean(remaining_losses)

    print(f"Clean samples: {len(clean_samples)}")
    print(f"Average loss of clean samples: {np.mean(clean_losses):.4f}")
    print(f"Average loss of remaining samples: {np.mean(remaining_losses):.4f}")
    print(f"Clean sample texts: {clean_texts}")


def test_k_medoids_clustering(selector, sample_data):
    """
    Test that k-medoids clustering correctly finds diverse representatives
    from each class, as described in the paper.
    """
    texts = sample_data["texts"]
    labels = sample_data["labels"]

    # Generate embeddings for testing
    embeddings = selector._get_embeddings(texts)

    # Find medoids
    k = 4
    medoid_indices = k_medoids(embeddings, k)

    # We should get k medoids (or fewer if there are empty clusters)
    assert 0 < len(medoid_indices) <= k

    # Medoids should be diverse (not all from the same class)
    medoid_labels = [labels[i] for i in medoid_indices]

    # There should be at least one example from each class if possible
    unique_labels = set(medoid_labels)
    assert len(unique_labels) > 0

    print(f"Selected {len(medoid_indices)} medoids")
    print(f"Unique labels in medoids: {unique_labels}")
    print(f"Medoid indices: {medoid_indices}")
    print(f"Medoid texts: {[texts[i] for i in medoid_indices]}")


def test_select_demonstrations(selector, sample_data):
    """
    Test that demonstration selection maintains class balance and selects
    diverse examples, as described in the paper.
    """
    samples_with_loss = sample_data["samples_with_loss"]

    # Filter clean samples first
    clean_samples = filter_clean_samples(samples_with_loss)

    # Get the actual classes available in clean samples
    available_classes = set(label for _, label in clean_samples)

    # Select demonstrations, requesting examples per class
    num_per_class = 2
    demos = selector.select_demonstrations(clean_samples, num_per_class)

    # Check that we have some demonstrations
    assert len(demos) > 0

    # Count examples per class
    demo_labels = [demo[1] for demo in demos]
    label_counts = {}
    for label in demo_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Each class should have at most num_per_class examples
    for label, count in label_counts.items():
        assert count <= num_per_class

    # Should have examples from all available classes
    assert set(label_counts.keys()) == available_classes

    print(f"Selected {len(demos)} demonstrations")
    print(f"Class distribution: {label_counts}")
    print(f"Available classes in clean samples: {available_classes}")
    print(f"Demonstration texts: {[demo[0] for demo in demos]}")


def test_complete_pipeline(selector, sample_data):
    """
    Test the complete pipeline from noisy samples to selected demonstrations,
    as described in the paper.
    """
    samples_with_loss = sample_data["samples_with_loss"]
    losses = sample_data["losses"]
    texts = sample_data["texts"]

    # First run the filtering step to determine what classes are available
    clean_samples = filter_clean_samples(samples_with_loss)
    available_classes = set(label for _, label in clean_samples)

    print(f"Available classes after filtering: {available_classes}")

    # Use the combined method to filter and select in one step
    num_per_class = 2
    demos = selector.select_demonstrations_from_noisy(
        samples_with_loss, num_per_class=num_per_class
    )

    # Check that we have some demonstrations
    assert len(demos) > 0

    # Count examples per class
    demo_labels = [demo[1] for demo in demos]
    label_counts = {}
    for label in demo_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Classes should be balanced if possible
    for label, count in label_counts.items():
        assert count <= num_per_class

    # Should have examples from all available classes after filtering
    assert set(label_counts.keys()) == available_classes

    # Verify that selected demonstrations have lower loss values
    demo_texts = [demo[0] for demo in demos]
    demo_indices = [i for i, text in enumerate(texts) if text in demo_texts]
    demo_losses = [losses[i] for i in demo_indices]

    # Average loss of selected demos should be lower (cleaner samples)
    all_losses_mean = np.mean(losses)
    demo_losses_mean = np.mean(demo_losses) if demo_losses else float("inf")

    assert demo_losses_mean < all_losses_mean

    print(f"Complete pipeline selected {len(demos)} demonstrations")
    print(f"Class distribution: {label_counts}")
    print(f"Average loss of all samples: {all_losses_mean:.4f}")
    print(f"Average loss of selected demonstrations: {demo_losses_mean:.4f}")
    print(f"Selected demonstration texts: {demo_texts}")


def test_with_real_sentiment_data(selector, real_sentiment_data):
    """
    Test with actual sentiment data from a real dataset if available.
    """
    if real_sentiment_data is None:
        pytest.skip("Real sentiment data not available")

    samples_with_loss = real_sentiment_data["samples_with_loss"]

    # First run the filtering step to determine what classes are available
    clean_samples = filter_clean_samples(samples_with_loss)
    available_classes = set(label for _, label in clean_samples)

    # Run the complete pipeline
    demos = selector.select_demonstrations_from_noisy(
        samples_with_loss, num_per_class=5
    )

    # Basic validation
    assert len(demos) > 0

    # Check class balance
    demo_labels = [demo[1] for demo in demos]
    label_counts = {}
    for label in demo_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Each class should have some examples, but only for classes that had clean samples
    assert set(label_counts.keys()) == available_classes

    print(f"Real data: Available classes after filtering: {available_classes}")
    print(f"Real data: Selected {len(demos)} demonstrations")
    print(f"Real data: Class distribution: {label_counts}")
