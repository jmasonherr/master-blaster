from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

from api_key import defalt_bert_model


def select_diverse_initial_samples(
        samples: List[Tuple[str, str]],
        num_per_class: int = 10,
        model_name: str = defalt_bert_model,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16
) -> List[Tuple[str, str]]:
    """
    Select a diverse set of samples for each class using unsupervised clustering. Is meant for use
    in first round of selection

    Args:
        samples: List of (text, label) pairs
        num_per_class: Number of samples to select per class
        model_name: Name of the embedding model to use
        device: Device to use for computation ('cuda', 'cpu', or None for auto-detection)
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for embedding computation

    Returns:
        List of (text, label) pairs representing diverse samples across classes
    """
    if not samples:
        return []

    # Initialize embedding model
    tokenizer, model, device, max_length = create_embedding_model(
        model_name=model_name,
        device=device,
        max_length=max_length
    )

    # Group samples by class
    samples_by_class: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for i, (text, label) in enumerate(samples):
        samples_by_class[label].append((i, text))

    diverse_samples = []

    # Process each class separately
    for label, class_samples in samples_by_class.items():
        # Get text samples for this class
        texts = [sample[1] for sample in class_samples]

        # If we have fewer samples than requested, use all of them
        if len(texts) <= num_per_class:
            diverse_samples.extend(
                [(samples[sample[0]][0], label) for sample in class_samples]
            )
            continue

        # Get embeddings for this class
        embeddings = get_embeddings(
            texts,
            tokenizer,
            model,
            device,
            max_length,
            batch_size
        )

        # Use k-medoids to select diverse examples
        medoid_indices = k_medoids(embeddings, num_per_class)

        # Map back to original indices
        # For example, if medoid_indices = [3, 27, 152], it means that the samples at positions 3, 27, and 152 in your
        # dataset are the most representative of their respective clusters.
        original_indices = [class_samples[i][0] for i in medoid_indices]

        # Add selected samples
        for idx in original_indices:
            text = samples[idx][0]
            diverse_samples.append((text, label))

    return diverse_samples


def filter_clean_samples(
    samples_with_loss: List[Tuple[str, str, float]], threshold: float = 0.7
) -> List[Tuple[str, str]]:
    """
    Filter clean samples using a Gaussian Mixture Model on loss values.

    Args:
        samples_with_loss: List of tuples (text, label, loss)
        threshold: Probability threshold for clean samples

    Returns:
        List of (text, label) pairs considered clean
    """
    if not samples_with_loss:
        return []

    # Extract losses
    losses = np.array([sample[2] for sample in samples_with_loss]).reshape(-1, 1)

    # Fit GMM with 2 components (clean and noisy)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(losses)

    # Get component with smaller mean (clean samples)
    clean_component = np.argmin(gmm.means_)

    # Get probabilities of belonging to the clean component
    probs = gmm.predict_proba(losses)[:, clean_component]

    # Filter samples based on threshold
    clean_samples = [
        (sample[0], sample[1])
        for i, sample in enumerate(samples_with_loss)
        if probs[i] >= threshold
    ]

    return clean_samples


def k_medoids(embeddings: np.ndarray, k: int) -> List[int]:
    """
    Implement k-medoids clustering to find representative samples.

    Args:
        embeddings: Matrix of embeddings with shape (n_samples, embedding_dim)
        k: Number of medoids to select

    Returns:
        Indices of selected medoids
    """
    if len(embeddings) <= k:
        # If we have fewer samples than requested medoids, return all indices
        return list(range(len(embeddings)))

    if k <= 0:
        return []

    # First use KMeans to get initial centroids
    kmeans = KMeans(n_clusters=min(k, len(embeddings)), random_state=42).fit(embeddings)
    centers = kmeans.cluster_centers_

    # Assign each point to nearest centroid
    distances = pairwise_distances(embeddings, centers)
    cluster_assignment = np.argmin(distances, axis=1)

    # Find medoid for each cluster (point closest to centroid)
    medoid_indices = []
    for i in range(min(k, len(embeddings))):
        if np.sum(cluster_assignment == i) == 0:
            continue  # Skip empty clusters

        cluster_points_indices = np.where(cluster_assignment == i)[0]
        cluster_points = embeddings[cluster_points_indices]
        center = centers[i].reshape(1, -1)

        # Find the point with minimum distance to center
        within_distances = pairwise_distances(cluster_points, center)
        if len(within_distances) > 0:  # Make sure we have points in this cluster
            medoid_idx = cluster_points_indices[np.argmin(within_distances)]
            medoid_indices.append(medoid_idx)

    return medoid_indices


# Create models for embeddings
def create_embedding_model(
    model_name: str,
    device: Optional[str] = None,
    max_length: int = 512,
) -> Tuple[Any, Any, str, int]:
    """
    Initialize a model for text embeddings.

    Args:
        model_name: Name of the model to use for text embeddings
        device: Device to use for computation ('cuda', 'cpu', or None for auto-detection)
        max_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (tokenizer, model, device, max_length)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    return tokenizer, model, device, max_length


def get_embeddings(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Get embeddings for a list of texts with batch processing.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer for the model
        model: Model for generating embeddings
        device: Device to run model on
        max_length: Maximum sequence length
        batch_size: Size of batches for processing

    Returns:
        Array of embeddings with shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])

    # Use base_model instead of the classifier model
    base_model = model.base_model  # Access the base transformer model

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = base_model(**inputs)
            # Use mean pooling to get a single vector per text
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.extend([emb for emb in batch_embeddings])

    return np.array(all_embeddings)


def select_demonstrations(
    clean_samples: List[Tuple[str, str]],
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
    num_per_class: int = 10,
) -> List[Tuple[str, str]]:
    """
    Select diverse demonstrations using k-medoids clustering.

    Args:
        clean_samples: List of (text, label) pairs
        tokenizer: Tokenizer for the model
        model: Model for generating embeddings
        device: Device to run model on
        max_length: Maximum sequence length
        num_per_class: Number of examples to select per class

    Returns:
        List of (text, label) pairs for demonstrations
    """
    if not clean_samples:
        return []

    # Group samples by class
    samples_by_class: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for i, (text, label) in enumerate(clean_samples):
        samples_by_class[label].append((i, text))

    selected_demonstrations = []

    # Process each class separately
    for label, samples in samples_by_class.items():
        if len(samples) <= num_per_class:
            # If we have fewer samples than requested, use all of them
            selected_demonstrations.extend(
                [(clean_samples[sample[0]][0], label) for sample in samples]
            )
            continue

        # Get text samples for this class
        texts = [sample[1] for sample in samples]

        # Get embeddings
        embeddings = get_embeddings(texts, tokenizer, model, device, max_length)

        # Use k-medoids to select diverse examples
        if len(texts) > num_per_class:
            medoid_indices = k_medoids(embeddings, num_per_class)

            # Get original indices
            original_indices = [samples[i][0] for i in medoid_indices]

            # Add selected samples
            for idx in original_indices:
                text = clean_samples[idx][0]
                selected_demonstrations.append((text, label))
        else:
            # If we don't have enough samples for clustering
            selected_demonstrations.extend(
                [(clean_samples[sample[0]][0], label) for sample in samples]
            )

    return selected_demonstrations


def select_demonstrations_from_noisy(
    samples_with_loss: List[Tuple[str, str, float]],
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
    num_per_class: int = 10,
    threshold: float = 0.7,
) -> List[Tuple[str, str]]:
    """
    Combined method to filter clean samples and select diverse demonstrations.

    Args:
        samples_with_loss: List of tuples (text, label, loss)
        tokenizer: Tokenizer for the model
        model: Model for generating embeddings
        device: Device to run model on
        max_length: Maximum sequence length
        num_per_class: Number of examples to select per class
        threshold: Probability threshold for clean samples

    Returns:
        List of (text, label) pairs for demonstrations
    """
    # First filter clean samples
    clean_samples = filter_clean_samples(samples_with_loss, threshold)

    # Then select diverse demonstrations
    return select_demonstrations(
        clean_samples, tokenizer, model, device, max_length, num_per_class
    )
