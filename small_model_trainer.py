import torch
import numpy as np
import traceback
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from tqdm import tqdm
from collections import namedtuple, defaultdict

from api_key import defalt_bert_model
from models.datasets import TextClassificationDataset

# Import the provided modifications module for data augmentation
import modifications
from sampler import select_demonstrations, get_embeddings, k_medoids

# Define named tuples for structured returns
@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: Any
    device: torch.device
    label_map: Dict[str, int]
    inv_label_map: Dict[int, str]

TrainingStats = namedtuple(
    "TrainingStats", ["training_losses", "validation_accuracies", "sample_losses"]
)
SampleLossInfo = namedtuple("SampleLossInfo", ["loss", "prob", "clean"])
ClassificationResult = namedtuple(
    "ClassificationResult", ["text", "label", "confidence"]
)


def create_dataloader(
    texts: List[str],
    label_indices: List[int],
    tokenizer: Any,
    max_length: int = 512,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the given texts and label indices.

    Args:
        texts: List of text examples
        label_indices: List of numeric label indices
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length for tokenization
        batch_size: Size of batches for processing
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = TextClassificationDataset(
        texts=texts,
        labels=label_indices,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def make_labels_right(labels: Any, label_map: Dict[str,int], inv_label_map: Dict[int, str]) -> (List[str], List[int]):
    # something keeps messing up the labels. better to just coerce them into the correct type
    if isinstance(labels, tuple):
        labels = list(labels)
    if not isinstance(labels, list):
        numeric_labels = labels.tolist()
        labels = [inv_label_map[i] for i in labels.tolist()]
    else:
        numeric_labels = [label_map[label] for label in labels]
    return labels, numeric_labels

# Model initialization function
def initialize_model(
    model_name: str,
    num_labels: int,
    label_names: List[str],
    device: Optional[str] = None,
    learning_rate: float = 2e-5,
) -> ModelBundle:
    """
    Initialize a model for text classification.

    Args:
        model_name: Name of the pre-trained HuggingFace model
        num_labels: Number of classes for classification
        label_names: List of human-readable label names
        device: Device to use (cuda or cpu). If None, auto-detects.
        learning_rate: Learning rate for optimizer

    Returns:
        ModelBundle containing model, tokenizer, device, and label mappings
    """
    # Set up device (GPU or CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Create mappings between label names and indices
    label_map = {name: i for i, name in enumerate(label_names)}
    inv_label_map = {i: name for i, name in enumerate(label_names)}

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)

    return ModelBundle(model=model, tokenizer=tokenizer, device=device, label_map=label_map, inv_label_map=inv_label_map)


# Mixup function
def apply_mixup(
    embeddings: torch.Tensor,
    label_tensor: torch.Tensor,
    num_labels: int,
    mixup_alpha: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation to embeddings and labels.

    Args:
        embeddings: Input embeddings of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        label_tensor: Label indices of shape [batch_size]
        num_labels: Number of classes for one-hot encoding
        mixup_alpha: Alpha parameter for Beta distribution in mixup

    Returns:
        Tuple of mixed embeddings and mixed one-hot labels
    """
    batch_size = embeddings.size(0)

    # Generate mixing coefficients from Beta distribution
    lam = np.random.beta(mixup_alpha, mixup_alpha, batch_size)

    # Handle different embedding shapes based on dimension
    is_3d = len(embeddings.shape) == 3

    # Reshape lambda for proper broadcasting
    if is_3d:
        # For 3D embeddings: [batch_size, seq_len, hidden_dim]
        lam_tensor = torch.tensor(
            lam, device=embeddings.device, dtype=embeddings.dtype
        ).view(batch_size, 1, 1)
    else:
        # For 2D embeddings: [batch_size, hidden_dim]
        lam_tensor = torch.tensor(
            lam, device=embeddings.device, dtype=embeddings.dtype
        ).view(batch_size, 1)

    # Create random permutation indices
    indices = torch.randperm(batch_size, device=embeddings.device)

    # Mix embeddings
    mixed_embeddings = lam_tensor * embeddings + (1 - lam_tensor) * embeddings[indices]

    # Convert labels to one-hot
    one_hot_labels = torch.zeros(batch_size, num_labels, device=embeddings.device)
    one_hot_labels.scatter_(1, label_tensor.unsqueeze(1), 1)

    # Mix labels with 1D lambda values
    lam_for_labels = torch.tensor(
        lam, device=embeddings.device, dtype=one_hot_labels.dtype
    ).view(batch_size, 1)
    mixed_labels = (
        lam_for_labels * one_hot_labels + (1 - lam_for_labels) * one_hot_labels[indices]
    )

    return mixed_embeddings, mixed_labels


# GMM filtering functions
def calculate_sample_losses(
    model: torch.nn.Module,
    texts: List[str],
    labels: List[int],
    model_bundle: ModelBundle,
    batch_size: int = 16,
    max_length: int = 512,
) -> List[Tuple[str, str, float]]:
    """
    Calculate loss for each sample.

    Args:
        model: Trained PyTorch model
        texts: List of text examples
        labels: List of string labels
        model_bundle: Bundle containing model components
        batch_size: Size of batches for processing
        max_length: Maximum sequence length for tokenization

    Returns:
        List of (text, label, loss) tuples
    """

    # Convert string labels to numeric indices
    lables, numeric_labels = make_labels_right(labels, model_bundle.label_map, model_bundle.inv_label_map)

    # Create dataloader
    dataloader = create_dataloader(
        texts, numeric_labels, model_bundle.tokenizer, max_length, batch_size, shuffle=False
    )

    model.eval()
    all_losses = []

    # Calculate loss for each sample
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating sample losses"):
            input_ids = batch["input_ids"].to(model_bundle.device)
            attention_mask = batch["attention_mask"].to(model_bundle.device)
            batch_labels = batch["label"].to(model_bundle.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )

            # Get per-sample losses
            individual_losses = torch.nn.functional.cross_entropy(
                outputs.logits, batch_labels, reduction="none"
            )

            all_losses.extend(individual_losses.cpu().numpy())

    # If we have fewer losses than samples (shouldn't happen normally),
    # pad with high loss values
    if len(all_losses) < len(texts):
        all_losses.extend([float("inf")] * (len(texts) - len(all_losses)))

    return [(text, label, loss) for text, label, loss in zip(texts, labels, all_losses)]


def identify_clean_samples(
    samples_with_loss: List[Tuple[str, str, float]], threshold: float = 0.7
) -> Dict[int, SampleLossInfo]:
    """
    Use GMM to identify clean vs. noisy samples.

    Args:
        samples_with_loss: List of (text, label, loss) tuples
        threshold: Probability threshold for clean samples

    Returns:
        Dictionary mapping sample indices to SampleLossInfo
    """
    # Extract losses
    losses = np.array([sample[2] for sample in samples_with_loss]).reshape(-1, 1)

    # Fit GMM with 2 components (clean and noisy)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(losses)

    # Identify clean component (the one with smaller mean loss)
    clean_component = np.argmin(gmm.means_)

    # Get probabilities of samples belonging to clean component
    probs = gmm.predict_proba(losses)[:, clean_component]

    # Store results as dictionary for easier lookup
    sample_losses = {}
    for i, (_, _, loss) in enumerate(samples_with_loss):
        prob = probs[i]
        is_clean = prob >= threshold
        sample_losses[i] = SampleLossInfo(loss=loss, prob=prob, clean=is_clean)

    return sample_losses


def get_clean_indices(
    batch_indices: torch.Tensor,
    sample_losses: Optional[Dict[int, SampleLossInfo]] = None,
) -> torch.Tensor:
    """
    Get indices of clean samples in the current batch based on GMM analysis.

    Args:
        batch_indices: Indices of samples in the current batch
        sample_losses: Dictionary of sample losses with clean flags

    Returns:
        Indices of clean samples within the batch
    """
    if sample_losses is None:
        # No filtering (return all indices)
        return torch.arange(len(batch_indices))

    # Create a boolean mask for clean samples
    clean_mask = torch.zeros(len(batch_indices), dtype=torch.bool)

    # Check each sample in the batch
    for i, idx in enumerate(batch_indices):
        idx_item = idx.item()
        # Mark as clean if it's in our sample_losses dict and flagged as clean
        if idx_item in sample_losses and sample_losses[idx_item].clean:
            clean_mask[i] = True

    # Convert boolean mask to indices
    clean_indices = torch.nonzero(clean_mask).squeeze(-1)
    return clean_indices


# Main training function
def train_robust_model(
    labeled_data: List[Tuple[str, str]],
    label_names: List[str],
    validation_data: Optional[List[Tuple[str, str]]] = None,
    model_name: str = defalt_bert_model,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    mixup_alpha: float = 4.0,
    consistency_weight: float = 1.0,
    warmup_steps: int = 0,
    gmm_threshold: float = 0.7,
    device: Optional[str] = None,
) -> Tuple[ModelBundle, TrainingStats]:
    """
    Train a robust text classification model with mixup, consistency regularization,
    and GMM-based sample filtering.

    Args:
        labeled_data: List of (text, label) pairs for training
        validation_data: Optional list of (text, label) pairs for validation
        model_name: Name of the pre-trained HuggingFace model
        label_names: List of label names (extracted from data if None)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        max_length: Maximum sequence length for tokenization
        mixup_alpha: Alpha parameter for Beta distribution in mixup
        consistency_weight: Weight for consistency regularization loss
        warmup_steps: Number of warmup steps for learning rate scheduler
        gmm_threshold: Probability threshold for considering a sample "clean"
        device: Device to use (cuda or cpu). If None, auto-detects.

    Returns:
        Tuple of (trained model, model bundle, training statistics)
    """
    # Extract texts and labels
    texts, labels = zip(*labeled_data)


    # Initialize model and components
    model_bundle = initialize_model(
        model_name=model_name,
        num_labels=len(label_names),
        label_names=label_names,
        device=device,
        learning_rate=learning_rate,
    )
    labels, numeric_labels = make_labels_right(labels,model_bundle.label_map, model_bundle.inv_label_map)

    model = model_bundle.model
    # Convert string labels to numeric indices

    # Create dataloader
    train_dataloader = create_dataloader(
        texts, numeric_labels, model_bundle.tokenizer, max_length, batch_size, shuffle=True
    )

    # Prepare optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Calculate total training steps for the scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Dictionary to track losses for each sample across epochs
    all_losses: Dict[int, List[float]] = {i: [] for i in range(len(labeled_data))}

    # Store training statistics
    training_losses = []
    validation_accuracies = []
    sample_losses = None

    print("Model architecture information:")
    print(f"Model type: {type(model)}")
    print(f"Base model type: {type(model.base_model)}")
    print(f"Classifier type: {type(model.classifier)}")

    # Check if there's a pooler
    if hasattr(model.base_model, "pooler"):
        print(f"Pooler type: {type(model.base_model.pooler)}")
    else:
        print("No pooler found in base model")

    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        # Process batches with progress bar
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            # Move batch tensors to the correct device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)

            # Clear gradients from previous batch
            optimizer.zero_grad()

            # Forward pass for standard cross-entropy loss
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=lbls
            )

            loss = outputs.loss
            logits = outputs.logits

            # Calculate indices of samples in the current batch
            batch_indices = batch_idx * batch_size + torch.arange(
                len(input_ids), device=device
            )
            # Ensure we don't go out of bounds
            batch_indices = batch_indices[batch_indices < len(labeled_data)]

            # Record individual sample losses for later GMM filtering
            with torch.no_grad():
                individual_losses = torch.nn.functional.cross_entropy(
                    logits, lbls, reduction="none"
                )

            # Store losses for each sample
            for i, sample_idx in enumerate(batch_indices):
                if i < len(individual_losses):
                    all_losses[sample_idx.item()].append(individual_losses[i].item())

            # Apply robust training techniques after the first epoch
            # First epoch is used to gather initial loss statistics
            if epoch > 0:
                # If in later epochs, apply GMM-based filtering, mixup, and consistency
                batch_size_current = input_ids.size(0)

                # Apply sample filtering based on previous epoch's losses
                if epoch > 1 and sample_losses is not None:
                    # Use GMM-based filtering from previous epochs to identify clean samples
                    clean_indices = get_clean_indices(batch_indices, sample_losses)

                    if len(clean_indices) > 0:
                        # Only use clean samples for standard loss calculation
                        clean_input_ids = input_ids[clean_indices]
                        clean_attention_mask = attention_mask[clean_indices]
                        clean_labels = lbls[clean_indices]

                        outputs = model(
                            input_ids=clean_input_ids,
                            attention_mask=clean_attention_mask,
                            labels=clean_labels,
                        )

                        loss = outputs.loss

                # Apply mixup on embeddings (use the first layer's embeddings)
                # Mixup requires at least 2 samples to interpolate between
                if batch_size_current > 1:
                    try:
                        print("Step 1: Extracting embeddings...")
                        with torch.no_grad():
                            embedding_layer = model.base_model.embeddings
                            embeddings = embedding_layer(input_ids)

                        print("Step 2: Applying mixup...")
                        mixed_embeddings, mixed_labels = apply_mixup(
                            embeddings, lbls, len(label_names), mixup_alpha
                        )

                        print("Step 3: Forward pass with mixed embeddings...")
                        # Use the correct parameter name 'inputs_embeds'
                        mixed_model_outputs = model(
                            inputs_embeds=mixed_embeddings,
                            attention_mask=attention_mask,
                        )

                        # Get the logits from the full model output
                        mixed_logits = mixed_model_outputs.logits

                        print("Step 4: Calculating mixup loss...")
                        mixup_loss = torch.mean(
                            torch.sum(
                                -mixed_labels * torch.log_softmax(mixed_logits, dim=-1),
                                dim=-1,
                            )
                        )
                        print(f"Mixup loss calculated successfully. {mixup_loss:.4f}")
                        print(f"Loss calculated successfully: {loss:.4f}")

                        print("Step 5: Adding to total loss...")
                        loss = loss + mixup_loss
                        print("Mixup process completed successfully")

                    except Exception as e:
                        print(f"Error during mixup: {e}")
                        traceback.print_exc()

                # Apply consistency regularization
                try:
                    # Store the original logits for consistency calculation
                    orig_logits = logits.detach().clone()

                    # Create augmented versions of the texts in this batch
                    batch_texts = [
                        texts[idx.item()]
                        for idx in batch_indices
                        if idx.item() < len(texts)
                    ]
                    augmented_texts = [
                        modifications.drop_word_text_augmentation(text)
                        for text in batch_texts
                    ]

                    if augmented_texts:  # Make sure we have texts to process
                        # Tokenize augmented texts
                        augmented_encodings = model_bundle.tokenizer(
                            augmented_texts,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt",
                        ).to(device)

                        # Forward pass for augmented texts
                        aug_outputs = model(
                            input_ids=augmented_encodings["input_ids"],
                            attention_mask=augmented_encodings["attention_mask"],
                        )

                        aug_logits = aug_outputs.logits

                        # Make sure dimensions match
                        if orig_logits.size(0) == aug_logits.size(0):
                            # KL divergence loss to measure difference between predictions
                            consistency_loss = torch.nn.functional.kl_div(
                                torch.log_softmax(aug_logits, dim=-1),
                                torch.softmax(orig_logits, dim=-1),
                                reduction="batchmean",
                            )

                            # Add weighted consistency loss to total loss
                            loss = loss + consistency_weight * consistency_loss
                        else:
                            print(
                                f"Skipping consistency loss - batch size mismatch: {orig_logits.size(0)} vs {aug_logits.size(0)}"
                            )

                except Exception as e:
                    print(f"Consistency regularization error: {e}")
                    traceback.print_exc()

            # Backward pass and optimization
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        training_losses.append(avg_epoch_loss)
        print(f"Average loss: {avg_epoch_loss:.4f}")

        # Evaluate on validation set if provided
        if validation_data:
            val_accuracy = evaluate_model(
                validation_data, model_bundle, batch_size, max_length
            )
            validation_accuracies.append(val_accuracy)
            print(f"Validation accuracy: {val_accuracy:.4f}")

        # After each epoch (except the last), recalculate sample losses for filtering
        if epoch < epochs - 1:
            samples_with_loss = calculate_sample_losses(
                model, texts, labels, model_bundle, batch_size, max_length
            )
            sample_losses = identify_clean_samples(samples_with_loss, gmm_threshold)

    # Calculate final sample losses
    final_samples_with_loss = calculate_sample_losses(
        model, texts, labels, model_bundle, batch_size, max_length
    )
    final_sample_losses = identify_clean_samples(final_samples_with_loss, gmm_threshold)

    # Collect training statistics
    stats = TrainingStats(
        training_losses=training_losses,
        validation_accuracies=validation_accuracies,
        sample_losses=final_sample_losses,
    )

    return model_bundle, stats


# Evaluation and prediction functions
def evaluate_model(
    eval_data: List[Tuple[str, str]],
    model_bundle: ModelBundle,
    batch_size: int = 16,
    max_length: int = 512,
) -> float:
    """
    Evaluate the model on the given data.

    Args:
        model: Trained PyTorch model
        eval_data: List of (text, label) pairs
        model_bundle: Bundle containing model components
        batch_size: Size of batches for processing
        max_length: Maximum sequence length for tokenization

    Returns:
        Accuracy on the evaluation data (0.0 to 1.0)
    """
    model = model_bundle.model
    if not eval_data:
        return 0.0

    # Extract texts and labels
    texts, labels = zip(*eval_data)

    # Convert string labels to numeric indices
    numeric_labels = [model_bundle.label_map[label] for label in labels]

    # Create dataloader
    eval_dataloader = create_dataloader(
        texts, numeric_labels, model_bundle.tokenizer, max_length, batch_size, shuffle=False
    )

    model.eval()
    correct = 0
    total = 0

    # Evaluate without gradient tracking
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(model_bundle.device)
            attention_mask = batch["attention_mask"].to(model_bundle.device)
            batch_labels = batch["label"].to(model_bundle.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predicted class (highest logit)
            predictions = torch.argmax(outputs.logits, dim=1)

            # Count correct predictions
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def predict_labels(
    model: torch.nn.Module,
    texts: List[str],
    model_bundle: ModelBundle,
    batch_size: int = 16,
    max_length: int = 512,
) -> List[str]:
    """
    Make predictions on new texts.

    Args:
        model: Trained PyTorch model
        texts: List of texts to predict
        model_bundle: Bundle containing model components
        batch_size: Size of batches for processing
        max_length: Maximum sequence length for tokenization

    Returns:
        List of predicted label names
    """

    if not texts:
        return []

    # Create a simple dataset without labels
    # Use first label as dummy (not used for prediction)
    first_label = next(iter(model_bundle.label_map.keys()))
    dummy_labels = [0] * len(texts)  # Use numeric dummy labels

    # Create dataloader
    dataloader = create_dataloader(
        texts, dummy_labels, model_bundle.tokenizer, max_length, batch_size, shuffle=False
    )

    model.eval()
    all_predictions = []

    # Make predictions without gradient tracking
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model_bundle.device)
            attention_mask = batch["attention_mask"].to(model_bundle.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predicted class indices
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # Convert indices back to label names
    predicted_labels = [model_bundle.inv_label_map[idx] for idx in all_predictions]
    return predicted_labels


def predict_with_confidence(
    model: torch.nn.Module,
    texts: List[str],
    model_bundle: ModelBundle,
    batch_size: int = 16,
    max_length: int = 512,
) -> List[Tuple[str, str, float]]:
    """
    Make predictions with confidence scores.

    Args:
        model: Trained PyTorch model
        texts: List of texts to predict
        model_bundle: Bundle containing model components
        batch_size: Size of batches for processing
        max_length: Maximum sequence length for tokenization

    Returns:
        List of (text, predicted_label, confidence) tuples
    """

    if not texts:
        return []

    # Create a simple dataset without labels
    # Use first label as dummy (not used for prediction)
    first_label = next(iter(model_bundle.label_map.keys()))
    dummy_labels = [0] * len(texts)  # Use numeric dummy labels

    # Create dataloader
    dataloader = create_dataloader(
        texts, dummy_labels, model_bundle.tokenizer, max_length, batch_size, shuffle=False
    )

    model.eval()
    all_predictions = []
    all_confidences = []

    # Make predictions without gradient tracking
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model_bundle.device)
            attention_mask = batch["attention_mask"].to(model_bundle.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get predicted class indices and their probabilities
            confidence_values, predictions = torch.max(probs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidence_values.cpu().numpy())

    # Create result tuples
    results = [
        (text, model_bundle.inv_label_map[pred], conf)
        for text, pred, conf in zip(texts, all_predictions, all_confidences)
    ]

    return results


def select_examples_for_prompting(
    labeled_data: List[Tuple[str, str]],
    model_bundle: ModelBundle,
    num_per_class: int = 5,
    strategy: str = "diverse",
) -> List[Tuple[str, str]]:
    """
    Select examples for prompting a language model.

    Args:
        labeled_data: List of (text, label) pairs
        model_bundle: Bundle containing model components
        num_per_class: Number of examples to select per class
        strategy: Selection strategy ("random", "diverse", "confident")

    Returns:
        List of selected (text, label) pairs
    """
    tokenizer, model, device, label_map, _ = model_bundle

    if not labeled_data:
        return []

    # Extract texts and labels
    texts, labels = zip(*labeled_data)

    # Get unique labels
    unique_labels = set(labels)

    if strategy == "diverse":
        # Group samples by class
        samples_by_class = defaultdict(list)
        for i, (text, label) in enumerate(labeled_data):
            samples_by_class[label].append((i, text))

        selected_examples = []
        max_length = 512  # Default max length

        # Process each class separately
        for label, samples in samples_by_class.items():
            if len(samples) <= num_per_class:
                # If we have fewer samples than requested, use all of them
                selected_examples.extend(
                    [(labeled_data[sample[0]][0], label) for sample in samples]
                )
                continue

            # Get text samples for this class
            texts_for_class = [sample[1] for sample in samples]

            # Get embeddings using the existing function
            embeddings = get_embeddings(
                texts_for_class, tokenizer, model, device, max_length
            )

            # Use existing k-medoids to select diverse examples
            medoid_indices = k_medoids(embeddings, num_per_class)

            # Get original indices
            original_indices = [samples[i][0] for i in medoid_indices]

            # Add selected samples
            for idx in original_indices:
                text = labeled_data[idx][0]
                selected_examples.append((text, label))

        return selected_examples

    elif strategy == "random":
        # Random selection
        selected_examples = []
        for label in unique_labels:
            class_examples = [(t, l) for t, l in labeled_data if l == label]

            if len(class_examples) <= num_per_class:
                selected_examples.extend(class_examples)
            else:
                selected_indices = np.random.choice(
                    len(class_examples), num_per_class, replace=False
                )
                selected_examples.extend([class_examples[i] for i in selected_indices])

        return selected_examples

    elif strategy == "confident":
        # Select examples with highest confidence scores
        selected_examples = []

        for label in unique_labels:
            class_examples = [
                (i, t, l) for i, (t, l) in enumerate(labeled_data) if l == label
            ]

            if len(class_examples) <= num_per_class:
                selected_examples.extend([(t, l) for _, t, l in class_examples])
                continue

            # Get model predictions
            class_texts = [t for _, t, _ in class_examples]

            with torch.no_grad():
                inputs = tokenizer(
                    class_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

                # Get confidence scores for the true class
                class_idx = label_map[label]
                confidence_scores = probs[:, class_idx].cpu().numpy()

            # Select examples with highest confidence
            top_indices = np.argsort(confidence_scores)[-num_per_class:]
            selected_examples.extend(
                [(class_examples[i][1], label) for i in top_indices]
            )

        return selected_examples

    else:
        # Default to diverse selection
        return select_demonstrations(
            labeled_data,
            tokenizer,
            model,
            device,
            512,  # Default max_length
            num_per_class,
        )
