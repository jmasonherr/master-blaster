import pandas as pd
from sklearn.model_selection import train_test_split
import datasets


def load_imdb_dataset(max_samples=None, test_size=0.2, seed=42):
    """
    Load the IMDb dataset for sentiment analysis.

    Args:
        max_samples (int, optional): Maximum number of samples to use per class
        test_size (float): Proportion of data to use for testing
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_texts, train_labels, val_texts, val_labels)
    """
    print("Loading IMDb dataset...")

    # Use the datasets library to download IMDb
    imdb = datasets.load_dataset("imdb")

    # Convert to pandas for easier manipulation
    train_df = pd.DataFrame(imdb["train"])
    test_df = pd.DataFrame(imdb["test"])

    # Combine train and test to resplit
    all_df = pd.concat([train_df, test_df])

    # Filter data by class
    negative_samples = all_df[all_df["label"] == 0]
    positive_samples = all_df[all_df["label"] == 1]

    # Limit samples if specified (using max_samples per class)
    if max_samples is not None:
        negative_samples = negative_samples.sample(
            min(max_samples, len(negative_samples)), random_state=seed
        )
        positive_samples = positive_samples.sample(
            min(max_samples, len(positive_samples)), random_state=seed
        )

    # Combine balanced samples
    balanced_df = pd.concat([negative_samples, positive_samples])

    # Map the labels (0=negative, 1=positive)
    balanced_df["label_text"] = balanced_df["label"].map({0: "negative", 1: "positive"})

    # Split into train and validation
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=test_size,
        stratify=balanced_df["label"],
        random_state=seed,
    )

    print(
        f"Dataset loaded: {len(train_df)} training examples, {len(val_df)} validation examples"
    )

    return (
        train_df["text"].tolist(),
        train_df["label_text"].tolist(),
        val_df["text"].tolist(),
        val_df["label_text"].tolist(),
    )
