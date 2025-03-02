from typing import List, Tuple, Dict, Optional, Any

import json

from freeal_datasets import load_imdb_dataset

from api_key import anthropic_api_key as aik
from api_key import default_model

import random


from llm_annotator import LLMAnnotator
from sample_selector import SampleSelector, filter_clean_samples
from slm_trainer import SLMTrainer

import pandas as pd

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import time
from datetime import datetime


class FreeALEvaluator:
    """
    Evaluator class for tracking and analyzing FreeAL performance.

    This class handles metrics collection, visualization, and comparison with baselines.
    It tracks performance over multiple rounds and generates comprehensive reports.
    """

    def __init__(self, log_dir="evaluation_results"):
        """
        Initialize the evaluator for tracking FreeAL performance.

        Args:
            log_dir (str): Directory to save evaluation results
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialize metrics storage
        self.metrics = {
            "rounds": [],
            "llm_accuracy": [],
            "slm_accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "clean_sample_ratio": [],
            "elapsed_time": [],
        }

        # Track performance by class
        self.class_metrics = {}

        # Store confusion matrices
        self.confusion_matrices = []

    def evaluate_round(self, round_num, freeal, labeled_data, ground_truth, start_time):
        """
        Evaluate performance after a FreeAL round.

        Args:
            round_num (int): Current round number
            freeal: The FreeAL instance
            labeled_data: Current labeled data (text, predicted_label)
            ground_truth: Dictionary of {text: true_label}
            start_time: Time when round started

        Returns:
            dict: Dictionary of computed metrics
        """
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Extract texts and predicted labels
        texts = [text for text, _ in labeled_data]
        predicted_labels = [label for _, label in labeled_data]

        # Get ground truth labels for these texts
        true_labels = [ground_truth.get(text, None) for text in texts]

        # Filter out examples without ground truth
        valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
        filtered_pred = [predicted_labels[i] for i in valid_indices]
        filtered_true = [true_labels[i] for i in valid_indices]

        if len(filtered_true) == 0:
            print("Warning: No examples with ground truth found for evaluation")
            return {}

        # Calculate metrics
        accuracy = accuracy_score(filtered_true, filtered_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_true, filtered_pred, average="weighted"
        )

        # Get class-specific metrics
        class_precision, class_recall, class_f1, class_support = (
            precision_recall_fscore_support(filtered_true, filtered_pred, average=None)
        )

        # Calculate clean sample ratio if available
        clean_ratio = 0
        if (
            hasattr(freeal, "slm_trainer")
            and hasattr(freeal.slm_trainer, "sample_losses")
            and freeal.slm_trainer.sample_losses
        ):
            clean_count = sum(
                1 for info in freeal.slm_trainer.sample_losses.values() if info["clean"]
            )
            total_count = len(freeal.slm_trainer.sample_losses)
            clean_ratio = clean_count / total_count if total_count > 0 else 0

        # Get LLM accuracy (using demonstration pool examples)
        llm_accuracy = 0
        if freeal.demonstration_pool:
            demo_texts = [text for text, _ in freeal.demonstration_pool]
            demo_pred = [label for _, label in freeal.demonstration_pool]
            demo_true = [ground_truth.get(text, None) for text in demo_texts]

            # Filter examples with ground truth
            valid_demo = [
                (pred, true)
                for pred, true in zip(demo_pred, demo_true)
                if true is not None
            ]
            if valid_demo:
                demo_pred = [pred for pred, _ in valid_demo]
                demo_true = [true for _, true in valid_demo]
                llm_accuracy = accuracy_score(demo_true, demo_pred)

        # Compute confusion matrix
        cm = confusion_matrix(filtered_true, filtered_pred, labels=freeal.label_names)

        # Store metrics
        self.metrics["rounds"].append(round_num)
        self.metrics["llm_accuracy"].append(llm_accuracy)
        self.metrics["slm_accuracy"].append(accuracy)
        self.metrics["precision"].append(precision)
        self.metrics["recall"].append(recall)
        self.metrics["f1"].append(f1)
        self.metrics["clean_sample_ratio"].append(clean_ratio)
        self.metrics["elapsed_time"].append(elapsed_time)

        # Store class metrics
        for i, label in enumerate(freeal.label_names):
            if label not in self.class_metrics:
                self.class_metrics[label] = {
                    "precision": [],
                    "recall": [],
                    "f1": [],
                    "support": [],
                }

            self.class_metrics[label]["precision"].append(class_precision[i])
            self.class_metrics[label]["recall"].append(class_recall[i])
            self.class_metrics[label]["f1"].append(class_f1[i])
            self.class_metrics[label]["support"].append(class_support[i])

        # Store confusion matrix
        self.confusion_matrices.append(
            {"round": round_num, "matrix": cm, "labels": freeal.label_names}
        )

        # Save results
        self._save_results()

        # Return metrics for current round
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "llm_accuracy": llm_accuracy,
            "clean_ratio": clean_ratio,
            "elapsed_time": elapsed_time,
        }

    def compare_with_baselines(self, freeal, labeled_data, ground_truth, baselines):
        """
        Compare FreeAL performance with baseline methods.

        Args:
            freeal: The FreeAL instance
            labeled_data: Final labeled data
            ground_truth: Dictionary of {text: true_label}
            baselines: Dictionary of {method_name: predictions}

        Returns:
            DataFrame: Comparison results
        """
        # Get texts and FreeAL predictions
        texts = [text for text, _ in labeled_data]
        freeal_pred = [label for _, label in labeled_data]

        # Get ground truth labels
        true_labels = [ground_truth.get(text, None) for text in texts]

        # Filter out examples without ground truth
        valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
        filtered_texts = [texts[i] for i in valid_indices]
        filtered_freeal = [freeal_pred[i] for i in valid_indices]
        filtered_true = [true_labels[i] for i in valid_indices]

        # Calculate FreeAL metrics
        freeal_accuracy = accuracy_score(filtered_true, filtered_freeal)
        freeal_precision, freeal_recall, freeal_f1, _ = precision_recall_fscore_support(
            filtered_true, filtered_freeal, average="weighted"
        )

        # Prepare results
        results = {
            "Method": ["FreeAL"],
            "Accuracy": [freeal_accuracy],
            "Precision": [freeal_precision],
            "Recall": [freeal_recall],
            "F1 Score": [freeal_f1],
        }

        # Calculate metrics for each baseline
        for method, predictions in baselines.items():
            # Get predictions for filtered texts
            filtered_baseline = [predictions.get(text, "") for text in filtered_texts]

            # Filter out missing predictions
            valid_baseline = [
                (pred, true)
                for pred, true in zip(filtered_baseline, filtered_true)
                if pred
            ]
            if not valid_baseline:
                continue

            baseline_pred = [pred for pred, _ in valid_baseline]
            baseline_true = [true for _, true in valid_baseline]

            # Calculate metrics
            baseline_acc = accuracy_score(baseline_true, baseline_pred)
            baseline_prec, baseline_rec, baseline_f1, _ = (
                precision_recall_fscore_support(
                    baseline_true, baseline_pred, average="weighted"
                )
            )

            # Add to results
            results["Method"].append(method)
            results["Accuracy"].append(baseline_acc)
            results["Precision"].append(baseline_prec)
            results["Recall"].append(baseline_rec)
            results["F1 Score"].append(baseline_f1)

        # Convert to DataFrame
        comparison_df = pd.DataFrame(results)

        # Save comparison results
        comparison_path = os.path.join(self.log_dir, "baseline_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        self._plot_baseline_comparison(comparison_df)

        return comparison_df

    def _save_results(self):
        """Save all evaluation results to files"""
        # Save overall metrics
        metrics_df = pd.DataFrame(self.metrics)
        metrics_path = os.path.join(self.log_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Save class metrics
        for label, metrics in self.class_metrics.items():
            class_df = pd.DataFrame(
                {
                    "round": self.metrics["rounds"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "support": metrics["support"],
                }
            )
            class_path = os.path.join(self.log_dir, f"class_{label}_metrics.csv")
            class_df.to_csv(class_path, index=False)

        # Generate and save plots
        self._plot_metrics()
        self._plot_class_metrics()
        self._plot_confusion_matrices()

    def _plot_metrics(self):
        """Plot overall performance metrics"""
        # Set style
        sns.set(style="whitegrid")

        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        # Plot accuracy
        axs[0, 0].plot(
            self.metrics["rounds"],
            self.metrics["slm_accuracy"],
            "o-",
            label="SLM Accuracy",
        )
        axs[0, 0].plot(
            self.metrics["rounds"],
            self.metrics["llm_accuracy"],
            "s--",
            label="LLM Accuracy",
        )
        axs[0, 0].set_title("Accuracy over Rounds")
        axs[0, 0].set_xlabel("Round")
        axs[0, 0].set_ylabel("Accuracy")
        axs[0, 0].legend()

        # Plot precision, recall, F1
        axs[0, 1].plot(
            self.metrics["rounds"], self.metrics["precision"], "o-", label="Precision"
        )
        axs[0, 1].plot(
            self.metrics["rounds"], self.metrics["recall"], "s-", label="Recall"
        )
        axs[0, 1].plot(
            self.metrics["rounds"], self.metrics["f1"], "^-", label="F1 Score"
        )
        axs[0, 1].set_title("Precision, Recall, F1 over Rounds")
        axs[0, 1].set_xlabel("Round")
        axs[0, 1].set_ylabel("Score")
        axs[0, 1].legend()

        # Plot clean sample ratio
        axs[1, 0].plot(self.metrics["rounds"], self.metrics["clean_sample_ratio"], "o-")
        axs[1, 0].set_title("Clean Sample Ratio over Rounds")
        axs[1, 0].set_xlabel("Round")
        axs[1, 0].set_ylabel("Ratio")

        # Plot time per round
        axs[1, 1].bar(self.metrics["rounds"], self.metrics["elapsed_time"])
        axs[1, 1].set_title("Time per Round")
        axs[1, 1].set_xlabel("Round")
        axs[1, 1].set_ylabel("Time (seconds)")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "overall_metrics.png"))
        plt.close()

    def _plot_class_metrics(self):
        """Plot per-class performance metrics"""
        # Set style
        sns.set(style="whitegrid")

        # Create a separate plot for each metric
        metrics = ["precision", "recall", "f1"]

        for metric in metrics:
            plt.figure(figsize=(10, 6))

            for label, class_metric in self.class_metrics.items():
                plt.plot(
                    self.metrics["rounds"], class_metric[metric], "o-", label=f"{label}"
                )

            plt.title(f"Per-Class {metric.capitalize()} over Rounds")
            plt.xlabel("Round")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(self.log_dir, f"class_{metric}.png"))
            plt.close()

    def _plot_confusion_matrices(self):
        """Plot confusion matrices for each round"""
        for cm_data in self.confusion_matrices:
            round_num = cm_data["round"]
            cm = cm_data["matrix"]
            labels = cm_data["labels"]

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.title(f"Confusion Matrix - Round {round_num}")
            plt.xlabel("Predicted")
            plt.ylabel("True")

            plt.savefig(
                os.path.join(self.log_dir, f"confusion_matrix_round_{round_num}.png")
            )
            plt.close()

    def _plot_baseline_comparison(self, comparison_df):
        """Plot comparison with baselines"""
        # Set style
        sns.set(style="whitegrid")

        # Plot accuracy comparison
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        x = np.arange(len(comparison_df["Method"]))

        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, metric in enumerate(metrics):
            plt.bar(
                x + i * bar_width,
                comparison_df[metric],
                width=bar_width,
                label=metric,
                color=colors[i],
            )

        plt.xlabel("Method")
        plt.ylabel("Score")
        plt.title("Performance Comparison with Baselines")
        plt.xticks(x + bar_width * 1.5, comparison_df["Method"])
        plt.legend()

        plt.savefig(os.path.join(self.log_dir, "baseline_comparison.png"))
        plt.close()


class FreeAL:
    """
    FreeAL (Free Active Learning) orchestrator class.

    This class implements the FreeAL algorithm, which combines:
    1. LLM-based annotation with few-shot learning
    2. Small Language Model (SLM) training on annotated data
    3. Sample selection to identify clean examples
    4. Iterative refinement of the demonstration pool

    The goal is to efficiently label data using LLMs and improve performance
    through iterative self-training.
    """

    def __init__(
        self,
        llm_annotator,
        slm_trainer,
        sample_selector,
        label_names: List[str],
        log_dir: str = "freeal_logs",
    ):
        """
        Initialize the FreeAL orchestrator.

        Args:
            llm_annotator: The LLMAnnotator instance
            slm_trainer: The SLMTrainer instance
            sample_selector: The SampleSelector instance
            label_names (List[str]): List of possible label names
            log_dir (str): Directory to save logs and checkpoints
        """
        self.llm_annotator = llm_annotator
        self.slm_trainer = slm_trainer
        self.sample_selector = sample_selector
        self.label_names = label_names
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Initialize state
        self.current_round = 0
        self.annotations = {}  # Store all annotations: {text: label}
        self.demonstration_pool = []  # Current demonstration pool

        # Track metrics
        self.metrics = {
            "rounds": [],
            "annotation_counts": [],
            "demo_pool_size": [],
            "slm_train_loss": [],
            "slm_val_accuracy": [],
        }

    def _log_state(self, round_metrics: Dict[str, Any] = None):
        """
        Log the current state of the FreeAL process.

        Args:
            round_metrics (Dict[str, Any]): Additional metrics to log for this round
        """
        # Update metrics
        self.metrics["rounds"].append(self.current_round)
        self.metrics["annotation_counts"].append(len(self.annotations))
        self.metrics["demo_pool_size"].append(len(self.demonstration_pool))

        if round_metrics:
            for key, value in round_metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)

        # Save metrics to file
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save annotations
        annotations_file = os.path.join(
            self.log_dir, f"annotations_round_{self.current_round}.json"
        )
        with open(annotations_file, "w") as f:
            json.dump(self.annotations, f, indent=2)

        # Save demonstration pool
        demo_file = os.path.join(self.log_dir, f"demos_round_{self.current_round}.json")
        with open(demo_file, "w") as f:
            json.dump(self.demonstration_pool, f, indent=2)

        print(f"Round {self.current_round} state logged to {self.log_dir}")

    def _sample_initial_data(
        self,
        initial_labeled_data: List[Tuple[str, str]],
    ):
        """
        Initialize the first round data.

        Args:
            initial_labeled_data (List[Tuple[str, str]]):  few-shot examples
        Returns:
            List[Tuple[str, str]]: Initial demonstration pool
        """

        print(
            f"Using {len(initial_labeled_data)} provided examples as initial demonstration pool"
        )
        self.demonstration_pool = initial_labeled_data

        # Add to annotations
        for text, label in initial_labeled_data:
            self.annotations[text] = label

        return initial_labeled_data

    def run_iteration(
        self,
        unlabeled_data: List[str],
        validation_data: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Run one iteration of the FreeAL algorithm.

        Args:
            unlabeled_data (List[str]): List of unlabeled text examples
            validation_data (Optional[List[Tuple[str, str]]]): Optional validation data

        Returns:
            Dict[str, Any]: Metrics and results from this iteration
        """
        print(f"=== Starting FreeAL Iteration {self.current_round + 1} ===")

        # Step 1: Use LLM for annotation with current demonstration pool
        to_annotate = []
        for text in unlabeled_data:
            if text not in self.annotations:
                to_annotate.append(text)

        print(f"Annotating {len(to_annotate)} examples with LLM...")

        # Process in smaller batches to avoid API rate limits
        batch_size = 50
        all_annotations = []

        for i in tqdm(range(0, len(to_annotate), batch_size)):
            batch = to_annotate[i : i + batch_size]
            annotations = self.llm_annotator.annotate_batch(
                batch, self.demonstration_pool
            )
            all_annotations.extend(annotations)

            # Small pause to avoid API rate limits
            time.sleep(1)

        # Update annotations
        for text, label in all_annotations:
            self.annotations[text] = label

        # Prepare data for SLM training
        labeled_data = [
            (text, self.annotations[text])
            for text in unlabeled_data
            if text in self.annotations
        ]

        # Step 2: Train SLM on the annotated data
        print(f"Training SLM on {len(labeled_data)} annotated examples...")
        self.slm_trainer.train(
            labeled_data=labeled_data, validation_data=validation_data, epochs=3
        )

        # Step 3: Calculate sample losses to identify clean samples
        print("Calculating sample losses...")
        samples_with_loss = self.slm_trainer.calculate_sample_losses(labeled_data)

        # Step 4: Filter clean samples and select new demonstration pool
        print("Selecting new demonstration pool...")
        clean_samples = filter_clean_samples(samples_with_loss)
        print(
            f"Filtered {len(clean_samples)} clean samples from {len(samples_with_loss)} total"
        )

        # Balance the demonstration pool across classes
        demonstrations_per_class = 10  # Select 10 demonstrations per class
        self.demonstration_pool = self.sample_selector.select_demonstrations(
            clean_samples, num_per_class=demonstrations_per_class
        )

        print(
            f"Selected {len(self.demonstration_pool)} diverse demonstrations for next round"
        )

        # Update metrics
        self.current_round += 1

        # Evaluate SLM if validation data is provided
        val_accuracy = None
        if validation_data:
            val_accuracy = self.slm_trainer.evaluate(validation_data)
            print(f"SLM Validation accuracy: {val_accuracy:.4f}")

        # Log state
        round_metrics = {
            "clean_samples": len(clean_samples),
            "total_samples": len(samples_with_loss),
            "val_accuracy": val_accuracy,
        }
        self._log_state(round_metrics)

        return {
            "round": self.current_round,
            "annotations": self.annotations,
            "demonstration_pool": self.demonstration_pool,
            "clean_samples": len(clean_samples),
            "val_accuracy": val_accuracy,
        }

    def run_full_loop(
        self,
        unlabeled_data: List[str],
        initial_labeled_data: List[Tuple[str, str]],
        validation_data: Optional[List[Tuple[str, str]]] = None,
        iterations: int = 3,
        convergence_threshold: float = 0.005,
    ):
        """
        Run the full FreeAL loop for multiple iterations.

        Args:
            unlabeled_data (List[str]): List of unlabeled text examples
            initial_labeled_data (List[Tuple[str, str]]): Few-shot examples
            validation_data (Optional[List[Tuple[str, str]]]): Optional validation data
            iterations (int): Maximum number of iterations
            convergence_threshold (float): Threshold for early stopping based on validation accuracy

        Returns:
            Dict[str, Any]: Results from the full training process
        """
        # Initialize with initial data
        self._sample_initial_data(initial_labeled_data)

        # Run iterations
        prev_accuracy = 0
        for i in range(iterations):
            print(f"\n--- FreeAL Round {i+1}/{iterations} ---\n")

            # Run one iteration
            results = self.run_iteration(
                unlabeled_data=unlabeled_data, validation_data=validation_data
            )

            # Check for convergence
            if validation_data and results["val_accuracy"] is not None:
                accuracy_gain = results["val_accuracy"] - prev_accuracy
                prev_accuracy = results["val_accuracy"]

                print(f"Accuracy gain: {accuracy_gain:.4f}")

                if i > 0 and accuracy_gain < convergence_threshold:
                    print(
                        f"Converged after {i+1} iterations (accuracy gain below threshold)"
                    )
                    break

        # Final evaluation
        if validation_data:
            final_accuracy = self.slm_trainer.evaluate(validation_data)
            print(f"\nFinal validation accuracy: {final_accuracy:.4f}")

        # Save final model
        model_path = os.path.join(self.log_dir, "final_model")
        os.makedirs(model_path, exist_ok=True)
        self.slm_trainer.model.save_pretrained(model_path)
        self.slm_trainer.tokenizer.save_pretrained(model_path)

        print(f"Final model saved to {model_path}")

        return {
            "rounds_completed": self.current_round,
            "final_annotations": self.annotations,
            "final_demonstration_pool": self.demonstration_pool,
            "final_accuracy": final_accuracy if validation_data else None,
            "model_path": model_path,
        }

    def predict(self, texts: List[str]):
        """
        Make predictions on new texts using the trained SLM.

        Args:
            texts (List[str]): List of texts to predict

        Returns:
            List[str]: Predicted label names
        """
        return self.slm_trainer.predict(texts)


# Modified FreeAL class with evaluation support
class FreeALWithEval(FreeAL):
    """
    Extended FreeAL class with built-in evaluation capabilities.

    This class adds evaluation functionality to the base FreeAL class,
    allowing for automatic performance tracking and comparison with baselines.
    """

    def __init__(self, *args, **kwargs):
        # Extract evaluator if provided
        self.evaluator = kwargs.pop("evaluator", None)
        self.ground_truth = kwargs.pop("ground_truth", {})

        # Initialize base class
        super().__init__(*args, **kwargs)

        # Create evaluator if not provided
        if self.evaluator is None:
            self.evaluator = FreeALEvaluator(
                log_dir=os.path.join(self.log_dir, "evaluation")
            )

    def run_iteration(self, *args, **kwargs):
        """Run one iteration with evaluation"""
        # Record start time
        start_time = time.time()

        # Run base iteration
        results = super().run_iteration(*args, **kwargs)

        # Evaluate if ground truth available
        if self.ground_truth:
            labeled_data = [(text, self.annotations[text]) for text in self.annotations]
            eval_results = self.evaluator.evaluate_round(
                self.current_round, self, labeled_data, self.ground_truth, start_time
            )

            # Add evaluation results
            results.update(eval_results)

        return results

    def run_zero_shot_baseline(self, unlabeled_data):
        """
        Run zero-shot baseline using LLM.

        Args:
            unlabeled_data (List[str]): List of unlabeled examples

        Returns:
            dict: Dictionary of {text: predicted_label}
        """
        print("Running zero-shot baseline...")

        # Clear demonstrations to ensure zero-shot
        temp_demonstrations = self.demonstration_pool
        self.demonstration_pool = []

        # Annotate in batches
        all_annotations = {}
        batch_size = 50

        for i in tqdm(range(0, len(unlabeled_data), batch_size)):
            batch = unlabeled_data[i : i + batch_size]
            annotations = self.llm_annotator.annotate_batch(batch)

            for text, label in annotations:
                all_annotations[text] = label

            # Small pause to avoid API rate limits
            time.sleep(1)

        # Restore demonstrations
        self.demonstration_pool = temp_demonstrations

        return all_annotations

    def run_few_shot_baseline(self, unlabeled_data, demonstrations):
        """
        Run few-shot baseline using LLM with fixed demonstrations.

        Args:
            unlabeled_data (List[str]): List of unlabeled examples
            demonstrations (List[Tuple[str, str]]): Demonstrations to use

        Returns:
            dict: Dictionary of {text: predicted_label}
        """
        print("Running few-shot baseline...")

        # Set fixed demonstrations
        temp_demonstrations = self.demonstration_pool
        self.demonstration_pool = demonstrations

        # Annotate in batches
        all_annotations = {}
        batch_size = 50

        for i in tqdm(range(0, len(unlabeled_data), batch_size)):
            batch = unlabeled_data[i : i + batch_size]
            annotations = self.llm_annotator.annotate_batch(batch, demonstrations)

            for text, label in annotations:
                all_annotations[text] = label

            # Small pause to avoid API rate limits
            time.sleep(1)

        # Restore demonstrations
        self.demonstration_pool = temp_demonstrations

        return all_annotations


def run_freeal_benchmark():
    """
    Run a comprehensive benchmark of FreeAL on the IMDb dataset.

    This function:
    1. Loads the IMDb dataset
    2. Sets up the FreeAL components (LLM annotator, SLM trainer, sample selector)
    3. Runs baseline methods (zero-shot and few-shot LLM)
    4. Runs the full FreeAL algorithm
    5. Compares performance and generates reports

    Returns:
        Tuple: (results, freeal_instance, comparison_dataframe)
    """
    # Configuration
    anthropic_api_key = aik

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"benchmarks/benchmark_imdb_{timestamp}"

    # Task details
    label_names = ["positive", "negative"]
    instruction = "Classify the sentiment of the following movie review as either positive or negative."
    task_description = """This is a sentiment analysis task for movie reviews. 
    Positive reviews express favorable opinions about the movie, while negative reviews express unfavorable opinions.
    Analyze the review text carefully and determine if the overall sentiment is positive or negative."""

    # Load data - use fewer samples for faster execution
    train_texts, train_labels, val_texts, val_labels = load_imdb_dataset(
        max_samples=2000
    )

    # Create validation data with ground truth labels
    validation_data = list(zip(val_texts, val_labels))

    # Create ground truth dictionary for all data
    ground_truth = {}
    for text, label in zip(train_texts, train_labels):
        ground_truth[text] = label
    for text, label in zip(val_texts, val_labels):
        ground_truth[text] = label

    # Create a few labeled examples for few-shot learning
    few_shot_indices = random.sample(range(len(train_texts)), 5)
    initial_examples = [(train_texts[i], train_labels[i]) for i in few_shot_indices]

    # Remove the few-shot examples from training data
    remaining_train_texts = [
        text for i, text in enumerate(train_texts) if i not in few_shot_indices
    ]

    print(
        f"Using {len(initial_examples)} labeled examples as few-shot learning examples"
    )
    print(f"Unlabeled training set: {len(remaining_train_texts)} examples")
    print(f"Validation set: {len(validation_data)} examples")

    # Initialize evaluator
    evaluator = FreeALEvaluator(log_dir=os.path.join(log_dir, "evaluation"))

    # Initialize components
    llm_annotator = LLMAnnotator(
        api_key=anthropic_api_key,
        instruction=instruction,
        task_description=task_description,
        label_names=label_names,
        examples=initial_examples[:5],
        model=default_model,
    )

    slm_trainer = SLMTrainer(
        model_name="distilroberta-base",
        num_labels=len(label_names),
        label_names=label_names,
        batch_size=16,
    )

    sample_selector = SampleSelector()

    freeal = FreeALWithEval(
        llm_annotator=llm_annotator,
        slm_trainer=slm_trainer,
        sample_selector=sample_selector,
        label_names=label_names,
        log_dir=log_dir,
        evaluator=evaluator,
        ground_truth=ground_truth,
    )

    # Run baselines before FreeAL
    print("\n=== Running Baselines ===\n")

    # Zero-shot baseline
    zero_shot_predictions = freeal.run_zero_shot_baseline(
        remaining_train_texts[:200]
    )  # Subset for speed

    # Few-shot baseline with fixed demonstrations
    few_shot_predictions = freeal.run_few_shot_baseline(
        remaining_train_texts[:200], initial_examples  # Subset for speed
    )

    # Run the full FreeAL loop
    print("\n=== Running FreeAL ===\n")
    results = freeal.run_full_loop(
        unlabeled_data=remaining_train_texts,
        initial_labeled_data=initial_examples,
        validation_data=validation_data,
        iterations=3,
    )

    # Compare FreeAL with baselines
    baselines = {
        "Zero-Shot LLM": zero_shot_predictions,
        "Few-Shot LLM": few_shot_predictions,
    }

    comparison = freeal.evaluator.compare_with_baselines(
        freeal,
        [(text, freeal.annotations.get(text, "")) for text in remaining_train_texts],
        ground_truth,
        baselines,
    )

    print("\n=== Baseline Comparison ===")
    print(comparison)

    # Evaluate on validation data
    val_predictions = freeal.predict(val_texts)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_labels, val_predictions, average="weighted"
    )

    print("\n=== Validation Performance ===")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")

    # Save validation metrics
    val_results = {
        "Accuracy": val_accuracy,
        "Precision": val_precision,
        "Recall": val_recall,
        "F1": val_f1,
    }

    with open(os.path.join(log_dir, "validation_results.json"), "w") as f:
        json.dump(val_results, f, indent=2)

    return results, freeal, comparison
