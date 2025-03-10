import os
import json
import time
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm

import api_key

# Import necessary functions from other modules
from batch_annotator import batch_classify
from collections import namedtuple
from small_model_trainer import (
    train_robust_model,
    evaluate_model,
    predict_labels,
    calculate_sample_losses,
)
from sampler import filter_clean_samples, select_demonstrations_from_noisy

# This function would need to be implemented based on how
# your models are saved and loaded in your framework
# For example with Hugging Face:
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class FreeALMetrics:
    """Metrics collected during FreeAL training"""

    rounds: List[int] = field(default_factory=list)
    annotation_counts: List[int] = field(default_factory=list)
    demo_pool_size: List[int] = field(default_factory=list)
    slm_val_accuracy: List[float] = field(default_factory=list)
    clean_samples_count: List[int] = field(default_factory=list)

    def save(self, filepath: str):
        """Save metrics to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(vars(self), f, indent=2)


@dataclass
class FreeALResult:
    """Results from FreeAL training process"""
    rounds_completed: int
    final_annotations: Dict[str, str]
    final_demonstration_pool: List[Tuple[str, str]]
    metrics: FreeALMetrics
    model: Any  # The trained model
    model_bundle: Any  # Model bundle containing model, tokenizer, etc.
    final_accuracy: Optional[float] = None
    model_path: str = ""  # Path where model is saved


def freeal(
    unlabeled_data: List[str],
    initial_examples: List[Tuple[str, str]],
    label_names: List[str],
    task_description: str,
    validation_data: Optional[List[Tuple[str, str]]] = None,
    model_name: str = api_key.defalt_bert_model,
    max_iterations: int = 4,
    log_dir: str = "freeal_logs",
    annotation_batch_size: int = 100,
    epochs_per_iteration: int = 3,
    demos_per_class: int = 10,
    convergence_threshold: float = 0.005,
    llm_model: str = api_key.default_model,
) -> FreeALResult:
    """
    Run the FreeAL (Free Active Learning) algorithm.

    This algorithm combines:
    1. LLM-based annotation with few-shot learning
    2. Small Language Model (SLM) training on annotated data
    3. Sample selection to identify clean examples
    4. Iterative refinement of the demonstration pool

    Args:
        unlabeled_data: List of unlabeled text examples
        initial_examples: Initial demonstration samples
        label_names: List of label names for the task
        task_description: Description of the classification task for the LLM
        validation_data: Optional validation data for evaluation
        model_name: Name of the pre-trained model to use for SLM
        max_iterations: Maximum number of iterations
        log_dir: Directory to save logs and checkpoints
        annotation_batch_size: Batch size for LLM annotation
        epochs_per_iteration: Number of training epochs per iteration
        demos_per_class: Number of demonstrations to select per class
        convergence_threshold: Threshold for early stopping
        llm_model: LLM model to use for annotation

    Returns:
        FreeALResult: Results from the training process including the trained model
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Initialize state
    current_round = 0
    annotations = {}  # Store all annotations: {text: label}
    demonstration_pool = initial_examples

    # Initialize metrics tracking
    metrics = FreeALMetrics()

    # Main training loop
    prev_accuracy = 0
    slm_model = None
    model_bundle = None

    for iteration in range(max_iterations):
        print(f"\n=== FreeAL Iteration {iteration + 1}/{max_iterations} ===\n")

        # ----- Step 1: Use LLM to annotate unlabeled data with current demonstrations -----
        to_annotate = [text for text in unlabeled_data if text not in annotations]
        print(f"Annotating {len(to_annotate)} examples with LLM...")

        # Process in smaller batches to avoid API rate limits
        all_annotations = []
        for i in tqdm(range(0, len(to_annotate), annotation_batch_size)):
            batch = to_annotate[i : i + annotation_batch_size]
            # Use the batch_classify function to get annotations from LLM
            batch_results = batch_classify(
                texts=batch,
                examples=demonstration_pool,
                label_names=label_names,
                task_description=task_description,
                model=llm_model,
            )
            # Extract text and matched_label from results
            batch_annotations = [
                (result.text, result.matched_label) for result in batch_results
            ]
            all_annotations.extend(batch_annotations)
            time.sleep(1)  # Small pause to avoid rate limits

        # Update annotations dictionary
        for text, label in all_annotations:
            annotations[text] = label

        # Prepare data for SLM training
        labeled_data = [
            (text, annotations[text]) for text in unlabeled_data if text in annotations
        ]

        # ----- Step 2: Train SLM on the annotated data -----
        print(f"Training SLM on {len(labeled_data)} annotated examples...")
        model_bundle, stats = train_robust_model(
            labeled_data=labeled_data,
            validation_data=validation_data,
            model_name=model_name,
            label_names=label_names,
            epochs=epochs_per_iteration,
        )

        # Save checkpoint of the model after each iteration
        checkpoint_dir = os.path.join(log_dir, f"checkpoint_round_{current_round + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_bundle.model.save_pretrained(checkpoint_dir)
        model_bundle.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved model checkpoint to {checkpoint_dir}")

        # ----- Step 3: Calculate sample losses to identify clean samples -----
        print("Calculating sample losses...")
        samples_with_loss = calculate_sample_losses(
            model=model_bundle.model,
            texts=[text for text, _ in labeled_data],
            labels=[label for _, label in labeled_data],
            model_bundle=model_bundle,
        )

        # ----- Step 4: Filter clean samples and select new demonstration pool -----
        print("Selecting new demonstration pool...")
        clean_samples = filter_clean_samples(samples_with_loss)
        print(
            f"Filtered {len(clean_samples)} clean samples from {len(samples_with_loss)} total"
        )

        # Select demonstrations for next round
        demonstration_pool = select_demonstrations_from_noisy(
            samples_with_loss=samples_with_loss,
            tokenizer=model_bundle.tokenizer,
            model=model_bundle.model,
            device=model_bundle.device,
            max_length=512,
            num_per_class=demos_per_class,
        )

        print(
            f"Selected {len(demonstration_pool)} diverse demonstrations for next round"
        )

        # ----- Evaluate and log results -----
        current_round += 1

        # Get validation accuracy if validation data is provided
        val_accuracy = None
        if validation_data:
            val_accuracy = evaluate_model(validation_data, model_bundle)
            print(f"SLM Validation accuracy: {val_accuracy:.4f}")
            metrics.slm_val_accuracy.append(val_accuracy)

            # Check for convergence
            if iteration > 0:
                accuracy_gain = val_accuracy - prev_accuracy
                print(f"Accuracy gain: {accuracy_gain:.4f}")

                if accuracy_gain < convergence_threshold:
                    print(
                        f"Converged after {iteration + 1} iterations (accuracy gain below threshold)"
                    )
                    break

            prev_accuracy = val_accuracy

        # Update metrics
        metrics.rounds.append(current_round)
        metrics.annotation_counts.append(len(annotations))
        metrics.demo_pool_size.append(len(demonstration_pool))
        metrics.clean_samples_count.append(len(clean_samples))

        # Save metrics to file
        metrics_file = os.path.join(log_dir, "metrics.json")
        metrics.save(metrics_file)

        # Save annotations and demonstration pool
        annotations_file = os.path.join(
            log_dir, f"annotations_round_{current_round}.json"
        )
        with open(annotations_file, "w") as f:
            json.dump({text: label for text, label in annotations.items()}, f, indent=2)

        demo_file = os.path.join(log_dir, f"demos_round_{current_round}.json")
        with open(demo_file, "w") as f:
            json.dump(demonstration_pool, f, indent=2)

    # ----- Finalize training -----
    # Final evaluation
    final_accuracy = None
    if validation_data and slm_model is not None:
        final_accuracy = evaluate_model(validation_data, model_bundle)
        print(f"\nFinal validation accuracy: {final_accuracy:.4f}")

    # Save final model
    model_path = os.path.join(log_dir, "final_model")
    os.makedirs(model_path, exist_ok=True)

    if model_bundle.model is not None:
        model_bundle.model.save_pretrained(model_path)
        model_bundle.tokenizer.save_pretrained(model_path)
        print(f"Final model saved to {model_path}")
    else:
        print("Warning: No model was trained. Cannot save final model.")

    # Return results
    return FreeALResult(
        rounds_completed=current_round,
        final_annotations=annotations,
        final_demonstration_pool=demonstration_pool,
        metrics=metrics,
        model=slm_model,
        model_bundle=model_bundle,
        final_accuracy=final_accuracy,
        model_path=model_path,
    )


def predict_with_freeal_model(model, model_bundle, texts: List[str]) -> List[str]:
    """
    Make predictions using a trained FreeAL model.

    Args:
        model: The trained model
        model_bundle: Model bundle containing tokenizer and label mappings
        texts: List of texts to predict

    Returns:
        List of predicted labels
    """
    return predict_labels(model, texts, model_bundle)


def load_freeal_model(model_path: str):
    """
    Load a saved FreeAL model from disk.

    Args:
        model_path: Path to the saved model directory

    Returns:
        Tuple of (model, model_bundle)
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Reconstruct model_bundle (this would depend on your specific implementation)
    # For a complete implementation, you'd need to load label mappings as well

    # This is a placeholder - actual implementation would depend on your model_bundle structure

    ModelBundle = namedtuple(
        "ModelBundle", ["model", "tokenizer", "device", "label_map", "inv_label_map"]
    )

    # Attempt to load label mappings from saved files

    label_map = {}
    inv_label_map = {}

    try:
        with open(os.path.join(model_path, "label_map.json"), "r") as f:
            label_map = json.load(f)
        with open(os.path.join(model_path, "inv_label_map.json"), "r") as f:
            inv_label_map = json.load(f)
    except FileNotFoundError:
        print("Warning: Label mappings not found. Using empty mappings.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_bundle = ModelBundle(model, tokenizer, device, label_map, inv_label_map)

    return model, model_bundle
