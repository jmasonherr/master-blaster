from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict


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


class SampleSelector:
    """
    A class to select clean and diverse samples as described in the FreeAL paper.

    This class implements:
    1. Filtering clean samples using a Gaussian Mixture Model on loss values
    2. Selecting diverse examples through k-medoids clustering
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize the sample selector with a sentence transformer model for embeddings.

        Args:
            model_name: Name of the model to use for text embeddings
            device: Device to use for computation ('cuda', 'cpu', or None for auto-detection)
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

    def _get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Get embeddings for a list of texts with batch processing.

        Args:
            texts: List of text strings
            batch_size: Size of batches for processing

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self.model(**inputs)
                # Use mean pooling to get a single vector per text
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.extend([emb for emb in batch_embeddings])

        return np.array(all_embeddings)

    def select_demonstrations(
        self, clean_samples: List[Tuple[str, str]], num_per_class: int = 10
    ) -> List[Tuple[str, str]]:
        """
        Select diverse demonstrations using k-medoids clustering.

        Args:
            clean_samples: List of (text, label) pairs
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
                    [(samples[i][1], label) for i in range(len(samples))]
                )
                continue

            # Get text samples for this class
            texts = [sample[1] for sample in samples]

            # Get embeddings
            embeddings = self._get_embeddings(texts)

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
                selected_demonstrations.extend([(text, label) for _, text in samples])

        return selected_demonstrations

    def select_demonstrations_from_noisy(
        self,
        samples_with_loss: List[Tuple[str, str, float]],
        num_per_class: int = 10,
        threshold: float = 0.7,
    ) -> List[Tuple[str, str]]:
        """
        Combined method to filter clean samples and select diverse demonstrations.

        Args:
            samples_with_loss: List of tuples (text, label, loss)
            num_per_class: Number of examples to select per class
            threshold: Probability threshold for clean samples

        Returns:
            List of (text, label) pairs for demonstrations
        """
        # First filter clean samples
        clean_samples = filter_clean_samples(samples_with_loss, threshold)

        # Then select diverse demonstrations
        return self.select_demonstrations(clean_samples, num_per_class)
