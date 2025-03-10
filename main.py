#!/usr/bin/env python
# freeal_imdb_cli.py

import argparse
import time

import torch
import random
import numpy as np
from typing import List, Tuple

from api_key import defalt_bert_model
# Import the FreeAL implementation
from free_al_core import freeal, predict_with_freeal_model, load_freeal_model

from freeal_datasets import load_imdb_dataset
from sampler import select_diverse_initial_samples


def set_seed(seed: int):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_initial_examples() -> List[Tuple[str, str]]:
    """Create a few initial examples for the sentiment classification task"""
    return [
        (
            "This movie was absolutely fantastic! I loved every minute of it.",
            "positive",
        ),
        ("The performances were outstanding and the story was engaging.", "positive"),
        ("I would highly recommend this film to anyone who enjoys drama.", "positive"),
        (
            "This was the worst movie I've ever seen. Complete waste of time.",
            "negative",
        ),
        ("The plot made no sense and the acting was terrible.", "negative"),
        ("I was so bored during this film that I almost fell asleep.", "negative"),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run FreeAL on IMDb sentiment dataset")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum samples per class (default: 500)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of FreeAL iterations (default: 4)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=defalt_bert_model,
        help="Base model name (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="freeal_imdb_results",
        help="Output directory for results (default: freeal_imdb_results)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for annotation (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--demos_per_class",
        type=int,
        default=10,
        help="Number of demonstrations per class (default: 10)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run predictions on test set after training",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load a previously trained model instead of training",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load dataset
    train_texts, train_labels, val_texts, val_labels = load_imdb_dataset(
        max_samples=args.max_samples, seed=args.seed
    )

    # Create validation data in the expected format
    validation_data = list(zip(val_texts, val_labels))

    # Task description for the LLM
    task_description = (
        "You are a sentiment analysis expert. Analyze the following movie review and classify it as either 'positive' or 'negative'"
    )

    # Label names for the task
    label_names = ["positive", "negative"]
    # takes about 15 seconds
    initial_examples = select_diverse_initial_samples(
        [x for x in zip(train_texts, train_labels)][:150],
        num_per_class=args.demos_per_class,
    )
    print("Initial examples:", len(initial_examples))
    print([f'{str(x)}\n' for x in initial_examples])


    # Create initial examples if training a new model
    if args.load_model is None:
        # Run FreeAL
        print(f"Starting FreeAL training with {args.iterations} iterations...")
        start_time = time.time()

        result = freeal(
            unlabeled_data=train_texts,
            initial_examples=initial_examples,
            label_names=label_names,
            task_description=task_description,
            validation_data=validation_data,
            model_name=args.model_name,
            max_iterations=args.iterations,
            log_dir=args.output_dir,
            annotation_batch_size=args.batch_size,
            demos_per_class=args.demos_per_class,
        )

        training_time = time.time() - start_time
        print(f"FreeAL training completed in {training_time:.2f} seconds")
        print(f"Final validation accuracy: {result.final_accuracy:.4f}")
        print(f"Model saved to: {result.model_path}")

        # Use the trained model and model_bundle from the result
        model = result.model
        model_bundle = result.model_bundle
    else:
        # Load a previously trained model
        print(f"Loading model from {args.load_model}...")
        model, model_bundle = load_freeal_model(args.load_model)
        print("Model loaded successfully")

    # Run predictions on test set if requested
    if args.predict:
        print("\nRunning predictions on validation set...")
        predictions = predict_with_freeal_model(model, model_bundle, val_texts[:5])

        # Print a few example predictions
        print("\nSample predictions:")
        for i, (text, label, pred) in enumerate(
            zip(val_texts[:5], val_labels[:5], predictions)
        ):
            match = "✓" if label == pred else "✗"
            print(f"Example {i + 1}:")
            print(f"Text: {text[:100]}...")
            print(f"True: {label}, Predicted: {pred} {match}\n")

        # Calculate accuracy on validation set
        all_predictions = predict_with_freeal_model(model, model_bundle, val_texts)
        accuracy = sum(
            1 for pred, label in zip(all_predictions, val_labels) if pred == label
        ) / len(val_labels)
        print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
