import torch
import numpy as np
import traceback
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Dict, Optional, Union, Any
from tqdm import tqdm

import modifications


class TextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification tasks.

    This dataset handles the conversion of raw text and labels into tensors
    that can be processed by the model.
    """

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int = 512
    ):
        """
        Dataset for text classification tasks.

        Args:
            texts (List[str]): List of text examples to classify
            labels (List[int]): List of numeric label indices (not label names)
            tokenizer: HuggingFace tokenizer for encoding texts
            max_length (int): Maximum sequence length for truncation/padding
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.

        Args:
            idx (int): Index of the example to retrieve

        Returns:
            Dict containing tokenized input_ids, attention_mask, and label tensor
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text and convert to PyTorch tensors
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension added by tokenizer
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }

        return item


class SLMTrainer:
    """
    Supervised Learning with Mixup (SLM) Trainer for robust text classification.

    This trainer implements several techniques to improve model robustness:
    1. Mixup augmentation - interpolates between examples
    2. Consistency regularization - ensures similar predictions for augmented inputs
    3. GMM-based sample filtering - identifies and focuses on "clean" samples
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        label_names: List[str],
        device: Optional[str] = None,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        mixup_alpha: float = 4.0,
        consistency_weight: float = 1.0,
    ):
        """
        Initialize the SLM Trainer for robust training.

        Args:
            model_name (str): Name of the pre-trained HuggingFace model
            num_labels (int): Number of classes for classification
            label_names (List[str]): List of human-readable label names
            device (str, optional): Device to use (cuda or cpu). If None, auto-detects.
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            max_length (int): Maximum sequence length for tokenization
            mixup_alpha (float): Alpha parameter for Beta distribution in mixup
                                 (higher values = less aggressive mixing)
            consistency_weight (float): Weight for consistency regularization loss
        """
        # Set up device (GPU or CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store configuration
        self.model_name = model_name
        self.num_labels = num_labels
        self.label_names = label_names

        # Create mappings between label names and indices
        self.label_map = {name: i for i, name in enumerate(label_names)}
        self.inv_label_map = {i: name for i, name in enumerate(label_names)}

        # Initialize tokenizer and model from pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.mixup_alpha = mixup_alpha
        self.consistency_weight = consistency_weight

        # Will store information about sample losses for filtering
        self.sample_losses: Optional[Dict[int, Dict[str, Union[float, bool]]]] = None

    def _create_dataloader(
        self, texts: List[str], labels: List[str], shuffle: bool = True
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for the given texts and labels.

        Args:
            texts (List[str]): List of text examples
            labels (List[str]): List of label names (will be converted to indices)
            shuffle (bool): Whether to shuffle the data

        Returns:
            DataLoader: PyTorch DataLoader for the dataset
        """
        # Convert string labels to numeric indices using the label map
        numeric_labels = [self.label_map[label] for label in labels]

        # Create dataset
        dataset = TextClassificationDataset(
            texts=texts,
            labels=numeric_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Create and return DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _mixup(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation to embeddings and labels.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape [batch_size, seq_len, hidden_dim]
                                      or [batch_size, hidden_dim]
            labels (torch.Tensor): Label indices of shape [batch_size]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mixed embeddings and mixed one-hot labels
        """
        batch_size = embeddings.size(0)

        # Generate mixing coefficients from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size)

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
        mixed_embeddings = (
            lam_tensor * embeddings + (1 - lam_tensor) * embeddings[indices]
        )

        # Convert labels to one-hot
        one_hot_labels = torch.zeros(
            batch_size, self.num_labels, device=embeddings.device
        )
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        # Mix labels with 1D lambda values
        lam_for_labels = torch.tensor(
            lam, device=embeddings.device, dtype=one_hot_labels.dtype
        ).view(batch_size, 1)

        mixed_labels = (
            lam_for_labels * one_hot_labels
            + (1 - lam_for_labels) * one_hot_labels[indices]
        )

        return mixed_embeddings, mixed_labels

    def train(
        self,
        labeled_data: List[Tuple[str, str]],
        validation_data: Optional[List[Tuple[str, str]]] = None,
        epochs: int = 5,
        warmup_steps: int = 0,
        gmm_threshold: float = 0.7,
    ) -> torch.nn.Module:
        """
        Train the model with robust techniques including mixup, consistency regularization,
        and GMM-based sample filtering.

        Args:
            labeled_data (List[Tuple[str, str]]): List of (text, label) pairs for training
            validation_data (Optional[List[Tuple[str, str]]]): Validation data for evaluation
            epochs (int): Number of training epochs
            warmup_steps (int): Number of warmup steps for learning rate scheduler
            gmm_threshold (float): Probability threshold for considering a sample "clean"

        Returns:
            The trained PyTorch model
        """
        # Unpack the labeled data into separate lists
        texts, labels = zip(*labeled_data)
        train_dataloader = self._create_dataloader(texts, labels, shuffle=True)

        # Prepare optimizer and learning rate scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        # Calculate total training steps for the scheduler
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Dictionary to track losses for each sample across epochs
        # Key: sample index, Value: list of loss values
        all_losses: Dict[int, List[float]] = {i: [] for i in range(len(labeled_data))}

        print("Model architecture information:")
        print(f"Model type: {type(self.model)}")
        print(f"Base model type: {type(self.model.base_model)}")
        print(f"Classifier type: {type(self.model.classifier)}")

        # Check if there's a pooler
        if hasattr(self.model.base_model, "pooler"):
            print(f"Pooler type: {type(self.model.base_model.pooler)}")
        else:
            print("No pooler found in base model")

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0

            # Process batches with progress bar
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                # Move batch tensors to the correct device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Clear gradients from previous batch
                optimizer.zero_grad()

                # Forward pass for standard cross-entropy loss
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # Calculate indices of samples in the current batch
                # This maps batch positions back to dataset indices
                batch_indices = batch_idx * self.batch_size + torch.arange(
                    len(input_ids), device=self.device
                )
                # Ensure we don't go out of bounds
                batch_indices = batch_indices[batch_indices < len(labeled_data)]

                # Record individual sample losses for later GMM filtering
                with torch.no_grad():
                    individual_losses = torch.nn.functional.cross_entropy(
                        logits, labels, reduction="none"  # Keep per-sample losses
                    )

                # Store losses for each sample
                for i, sample_idx in enumerate(batch_indices):
                    if i < len(individual_losses):
                        all_losses[sample_idx.item()].append(
                            individual_losses[i].item()
                        )

                # Apply robust training techniques after the first epoch
                # First epoch is used to gather initial loss statistics
                if epoch > 0:
                    # If in later epochs, apply GMM-based filtering, mixup, and consistency
                    batch_size = input_ids.size(0)

                    # Apply sample filtering based on previous epoch's losses
                    if epoch > 1:
                        # Use GMM-based filtering from previous epochs to identify clean samples
                        clean_indices = self._get_clean_indices(batch_indices)

                        if len(clean_indices) > 0:
                            # Only use clean samples for standard loss calculation
                            # This helps prevent noisy samples from affecting training
                            clean_input_ids = input_ids[clean_indices]
                            clean_attention_mask = attention_mask[clean_indices]
                            clean_labels = labels[clean_indices]

                            outputs = self.model(
                                input_ids=clean_input_ids,
                                attention_mask=clean_attention_mask,
                                labels=clean_labels,
                            )

                            loss = outputs.loss

                    # Apply mixup on embeddings (use the first layer's embeddings)
                    # Mixup requires at least 2 samples to interpolate between
                    # Apply mixup on embeddings (use the first layer's embeddings)
                    # Mixup requires at least 2 samples to interpolate between
                    if batch_size > 1:
                        try:
                            print("Step 1: Extracting embeddings...")
                            with torch.no_grad():
                                embedding_layer = self.model.base_model.embeddings
                                embeddings = embedding_layer(input_ids)

                            print("Step 2: Applying mixup...")
                            mixed_embeddings, mixed_labels = self._mixup(
                                embeddings, labels
                            )

                            print("Step 3: Forward pass with mixed embeddings...")
                            # Use the correct parameter name 'inputs_embeds' instead of trying to replace the embedding layer
                            mixed_model_outputs = self.model(
                                inputs_embeds=mixed_embeddings,
                                attention_mask=attention_mask,
                            )

                            # Get the logits from the full model output
                            mixed_logits = mixed_model_outputs.logits

                            print("Step 4: Calculating mixup loss...")
                            mixup_loss = torch.mean(
                                torch.sum(
                                    -mixed_labels
                                    * torch.log_softmax(mixed_logits, dim=-1),
                                    dim=-1,
                                )
                            )
                            print("Mixup loss calculated successfully")

                            print("Step 5: Adding to total loss...")
                            loss = loss + mixup_loss
                            print("Mixup process completed successfully")

                        except Exception as e:
                            print(f"Error during mixup: {e}")
                            import traceback

                            traceback.print_exc()
                    # Apply consistency regularization
                    # This encourages the model to make similar predictions for
                    # slightly different versions of the same input
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
                            augmented_encodings = self.tokenizer(
                                augmented_texts,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt",
                            ).to(self.device)

                            # Forward pass for augmented texts
                            aug_outputs = self.model(
                                input_ids=augmented_encodings["input_ids"],
                                attention_mask=augmented_encodings["attention_mask"],
                            )

                            aug_logits = aug_outputs.logits

                            # Make sure dimensions match - orig_logits might have different batch size
                            if orig_logits.size(0) == aug_logits.size(0):
                                # KL divergence loss to measure difference between predictions
                                consistency_loss = torch.nn.functional.kl_div(
                                    torch.log_softmax(aug_logits, dim=-1),
                                    torch.softmax(orig_logits, dim=-1),
                                    reduction="batchmean",
                                )

                                # Add weighted consistency loss to total loss
                                loss = loss + self.consistency_weight * consistency_loss
                            else:
                                print(
                                    f"Skipping consistency loss - batch size mismatch: {orig_logits.size(0)} vs {aug_logits.size(0)}"
                                )

                    except Exception as e:
                        print(f"Consistency regularization error: {e}")
                        traceback.print_exc()  # This will print the full traceback for debugging
                # Backward pass and optimization
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Average loss: {avg_epoch_loss:.4f}")

            # Evaluate on validation set if provided
            if validation_data:
                val_accuracy = self.evaluate(validation_data)
                print(f"Validation accuracy: {val_accuracy:.4f}")

        # After training, calculate final sample losses for filtering
        # This can be used for data cleaning in subsequent runs
        self.calculate_sample_losses(labeled_data)

        return self.model

    def _get_clean_indices(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Get indices of clean samples in the current batch based on previous GMM analysis.

        Clean samples are those that are likely to have correct labels according to
        the GMM model's probability estimates.

        Args:
            batch_indices (torch.Tensor): Indices of samples in the current batch

        Returns:
            torch.Tensor: Indices of clean samples within the batch
        """
        if self.sample_losses is None:
            # No filtering in the first epochs (return all indices)
            return torch.arange(len(batch_indices))

        # Create a boolean mask for clean samples
        clean_mask = torch.zeros(len(batch_indices), dtype=torch.bool)

        # Check each sample in the batch
        for i, idx in enumerate(batch_indices):
            # Mark as clean if it's in our sample_losses dict and flagged as clean
            if (
                idx.item() in self.sample_losses
                and self.sample_losses[idx.item()]["clean"]
            ):
                clean_mask[i] = True

        # Convert boolean mask to indices
        clean_indices = torch.nonzero(clean_mask).squeeze(-1)
        return clean_indices

    def calculate_sample_losses(
        self, labeled_data: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, float]]:
        """
        Calculate loss for each sample and use GMM to identify clean vs. noisy samples.

        This method:
        1. Calculates loss for each training sample
        2. Fits a 2-component Gaussian Mixture Model to the losses
        3. Identifies the "clean" component (typically lower mean loss)
        4. Assigns a probability of being "clean" to each sample

        Args:
            labeled_data (List[Tuple[str, str]]): List of (text, label) pairs

        Returns:
            List[Tuple[str, str, float]]: List of (text, label, loss) tuples
        """
        texts, labels = zip(*labeled_data)
        dataloader = self._create_dataloader(texts, labels, shuffle=False)

        self.model.eval()
        all_losses = []

        # Calculate loss for each sample
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating sample losses"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                outputs = self.model(
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
        if len(all_losses) < len(labeled_data):
            all_losses.extend([float("inf")] * (len(labeled_data) - len(all_losses)))

        # Fit a 2-component GMM to identify clean vs. noisy samples
        # The assumption is that clean samples have lower, more consistent losses
        # while noisy samples have higher, more variable losses
        losses_array = np.array(all_losses).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(losses_array)

        # Identify clean component (the one with smaller mean loss)
        clean_component = np.argmin(gmm.means_)

        # Get probabilities of samples belonging to clean component
        probs = gmm.predict_proba(losses_array)[:, clean_component]

        # Store results as dictionary for easier lookup
        self.sample_losses = {}

        # Convert to list of (text, label, loss) tuples with clean flag
        result = []
        for i, ((text, label), loss, prob) in enumerate(
            zip(labeled_data, all_losses, probs)
        ):
            is_clean = prob >= 0.7  # Use threshold to determine clean samples
            result.append((text, label, loss))
            self.sample_losses[i] = {"loss": loss, "prob": prob, "clean": is_clean}

        return result

    def evaluate(self, eval_data: List[Tuple[str, str]]) -> float:
        """
        Evaluate the model on the given data.

        Args:
            eval_data (List[Tuple[str, str]]): List of (text, label) pairs

        Returns:
            float: Accuracy on the evaluation data (0.0 to 1.0)
        """
        texts, labels = zip(*eval_data)
        eval_dataloader = self._create_dataloader(texts, labels, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0

        # Evaluate without gradient tracking
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get predicted class (highest logit)
                predictions = torch.argmax(outputs.logits, dim=1)

                # Count correct predictions
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)

        # Calculate accuracy
        accuracy = correct / total
        return accuracy

    def predict(self, texts: List[str]) -> List[str]:
        """
        Make predictions on new texts.

        Args:
            texts (List[str]): List of texts to predict

        Returns:
            List[str]: Predicted label names (not indices)
        """
        # Create a simple dataset without labels
        all_labels = list(self.label_map.keys())
        if len(all_labels) == 0:
            raise ValueError("No labels available")
        dummy_labels = [all_labels[0]] * len(
            texts
        )  # Dummy labels, not used for prediction
        dataloader = self._create_dataloader(texts, dummy_labels, shuffle=False)

        self.model.eval()
        all_predictions = []

        # Make predictions without gradient tracking
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get predicted class indices
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
        return all_predictions
