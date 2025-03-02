import os
import argparse
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

from freeal_datasets import load_imdb_dataset
from api_key import anthropic_api_key
from api_key import default_model

from llm_annotator import LLMAnnotator
from sample_selector import SampleSelector
from slm_trainer import SLMTrainer
from free_al import FreeALWithEval


def run_experiment(
    dataset_name="imdb", use_cache=True, max_samples=1000, iterations=3, output_dir=None
):
    """
    Run a FreeAL experiment on a specified dataset.

    Args:
        dataset_name (str): Name of the dataset to use
        use_cache (bool): Whether to use LLM caching
        max_samples (int): Maximum number of samples to use
        iterations (int): Number of FreeAL iterations to run
        output_dir (str): Directory to save results (defaults to timestamped directory)

    Returns:
        tuple: (results, freeal_instance, comparison_dataframe)
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir or f"experiments/{dataset_name}_{timestamp}"

    print(f"Running FreeAL experiment on {dataset_name} dataset")
    print(f"Results will be saved to {log_dir}")

    # Task details - adjust these based on the dataset
    if dataset_name.lower() == "imdb":
        label_names = ["positive", "negative"]
        instruction = "Classify the sentiment of the following movie review as either positive or negative."
        task_description = """This is a sentiment analysis task for movie reviews. 
        Positive reviews express favorable opinions about the movie, while negative reviews express unfavorable opinions.
        Analyze the review text carefully and determine if the overall sentiment is positive or negative."""

        # Load data
        train_texts, train_labels, val_texts, val_labels = load_imdb_dataset(
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    # Create validation data with ground truth labels
    validation_data = list(zip(val_texts, val_labels))

    # Create ground truth dictionary for all data
    ground_truth = {}
    for text, label in zip(train_texts, train_labels):
        ground_truth[text] = label
    for text, label in zip(val_texts, val_labels):
        ground_truth[text] = label

    # Create a few labeled examples for few-shot learning
    import random

    random.seed(42)  # For reproducibility
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
    from free_al import FreeALEvaluator

    evaluator = FreeALEvaluator(log_dir=os.path.join(log_dir, "evaluation"))

    # Initialize components
    llm_annotator = LLMAnnotator(
        api_key=anthropic_api_key,
        instruction=instruction,
        task_description=task_description,
        label_names=label_names,
        examples=initial_examples[:5],
        model=default_model,
        use_cache=use_cache,
        cache_path=os.path.join(log_dir, "llm_cache.db"),
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
    print("Running zero-shot baseline...")
    zero_shot_predictions = freeal.run_zero_shot_baseline(
        remaining_train_texts[:200]  # Subset for speed
    )

    # Few-shot baseline with fixed demonstrations
    print("Running few-shot baseline...")
    few_shot_predictions = freeal.run_few_shot_baseline(
        remaining_train_texts[:200], initial_examples  # Subset for speed
    )

    # Run the full FreeAL loop
    print("\n=== Running FreeAL ===\n")
    start_time = time.time()
    results = freeal.run_full_loop(
        unlabeled_data=remaining_train_texts,
        initial_labeled_data=initial_examples,
        validation_data=validation_data,
        iterations=iterations,
    )
    total_time = time.time() - start_time

    print(f"\nFreeAL completed in {total_time:.2f} seconds")

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

    # Save cache statistics if cache was used
    if use_cache:
        cache_stats = llm_annotator.get_cache_stats()
        with open(os.path.join(log_dir, "cache_stats.json"), "w") as f:
            json.dump(cache_stats, f, indent=2)

    # Create additional visualizations
    create_round_comparison_plots(freeal, log_dir)

    return results, freeal, comparison


def create_round_comparison_plots(freeal, log_dir):
    """Create additional visualization plots showing improvement by round"""
    # Make sure the plots directory exists
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load metrics from the evaluator
    metrics = freeal.evaluator.metrics

    # Set plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Plot all metrics in one figure
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rounds"], metrics["slm_accuracy"], "o-", label="SLM Accuracy")
    plt.plot(metrics["rounds"], metrics["llm_accuracy"], "s--", label="LLM Accuracy")
    plt.title("Accuracy by Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(metrics["rounds"], metrics["precision"], "o-", label="Precision")
    plt.plot(metrics["rounds"], metrics["recall"], "s-", label="Recall")
    plt.plot(metrics["rounds"], metrics["f1"], "^-", label="F1 Score")
    plt.title("Precision, Recall, and F1 by Round")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(metrics["rounds"], metrics["clean_sample_ratio"], "o-")
    plt.title("Clean Sample Ratio by Round")
    plt.xlabel("Round")
    plt.ylabel("Ratio")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.bar(metrics["rounds"], metrics["elapsed_time"])
    plt.title("Time per Round")
    plt.xlabel("Round")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "round_metrics.png"))
    plt.close()

    # Create a summary dataframe for additional plots
    summary_data = {
        "Round": ["Initial"] + [f"Round {r}" for r in metrics["rounds"]],
        "SLM Accuracy": [None] + metrics["slm_accuracy"],
        "LLM Accuracy": [
            metrics["llm_accuracy"][0] if metrics["llm_accuracy"] else None
        ]
        + metrics["llm_accuracy"],
        "F1 Score": [None] + metrics["f1"],
    }

    # Add zero-shot performance
    if hasattr(freeal, "zero_shot_accuracy"):
        summary_data["Round"][0] = "Zero-Shot"
        summary_data["LLM Accuracy"][0] = freeal.zero_shot_accuracy

    df = pd.DataFrame(summary_data)

    # Plot progression in a single chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df.melt(
            id_vars=["Round"],
            value_vars=["SLM Accuracy", "LLM Accuracy", "F1 Score"],
            var_name="Metric",
            value_name="Value",
        ),
        x="Round",
        y="Value",
        hue="Metric",
        marker="o",
    )
    plt.title("FreeAL Performance Progression")
    plt.ylim(0.5, 1.0)  # Adjust based on your data
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "performance_progression.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FreeAL experiments")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset to use")
    parser.add_argument(
        "--samples", type=int, default=35, help="Maximum samples to use"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of FreeAL iterations"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )

    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        use_cache=True,
        max_samples=args.samples,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
