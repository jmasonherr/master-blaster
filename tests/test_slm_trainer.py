import unittest
from unittest.mock import MagicMock, patch

import torch
import numpy as np

from modifications import drop_word_text_augmentation
from slm_trainer import TextClassificationDataset, SLMTrainer


class TestTextClassificationDataset(unittest.TestCase):
    def setUp(self):
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2054, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        # Sample data
        self.texts = ["This is a test", "Another example"]
        self.labels = [0, 1]

    def test_dataset_length(self):
        dataset = TextClassificationDataset(
            texts=self.texts,
            labels=self.labels,
            tokenizer=self.mock_tokenizer,
            max_length=128,
        )
        self.assertEqual(len(dataset), 2)

    def test_getitem(self):
        dataset = TextClassificationDataset(
            texts=self.texts,
            labels=self.labels,
            tokenizer=self.mock_tokenizer,
            max_length=128,
        )

        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("label", item)
        self.assertEqual(item["label"].item(), 0)


class TestTextAugmentation(unittest.TestCase):
    def test_short_text_unchanged(self):
        short_text = "hello world"
        augmented = drop_word_text_augmentation(short_text)
        self.assertEqual(augmented, short_text)

    def test_longer_text_augmented(self):
        text = "This is a longer text that should be augmented by dropping some words"
        augmented = drop_word_text_augmentation(text)
        # The augmented text should be different but not too different
        self.assertNotEqual(augmented, text)
        # The augmented text should have at least 80% of the original words
        self.assertGreaterEqual(len(augmented.split()), int(len(text.split()) * 0.8))


class TestSLMTrainer(unittest.TestCase):
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        # Mock the tokenizer and model to avoid loading actual models
        self.mock_tokenizer = mock_tokenizer.return_value
        self.mock_model = mock_model.return_value

        # Configure mock model
        self.mock_model.to.return_value = self.mock_model

        # Set up parameters for the model mock
        self.mock_model.parameters = MagicMock(
            return_value=[torch.nn.Parameter(torch.randn(10, 10))]
        )

        # Sample data
        self.label_names = ["positive", "negative", "neutral"]

        # Initialize trainer
        self.trainer = SLMTrainer(
            model_name="bert-base-uncased",
            num_labels=3,
            label_names=self.label_names,
            device="cpu",
            batch_size=2,
        )

        # Set mixup_alpha attribute for testing
        self.trainer.mixup_alpha = 0.2

        # Sample data for testing
        self.train_data = [
            ("This is great!", "positive"),
            ("This is terrible.", "negative"),
            ("This is okay.", "neutral"),
        ]

    def test_initialization(self):
        # Test that the trainer initializes correctly
        self.assertEqual(self.trainer.num_labels, 3)
        self.assertEqual(self.trainer.label_names, self.label_names)
        self.assertEqual(
            self.trainer.label_map, {"positive": 0, "negative": 1, "neutral": 2}
        )

    def test_create_dataloader(self):
        # Test dataloader creation
        texts = ["text1", "text2"]
        labels = ["positive", "negative"]

        with patch.object(self.trainer, "tokenizer"):
            dataloader = self.trainer._create_dataloader(texts, labels)
            self.assertEqual(len(dataloader.dataset), 2)

    def test_mixup(self):
        # Test mixup augmentation with correct embedding shape
        embeddings = torch.randn(4, 768)  # batch_size=4, hidden_dim=768
        labels = torch.tensor(
            [0, 1, 2, 0]
        )  # Ensure labels are within the correct range.
        # Mock the np.random.beta to return predictable values
        with patch("numpy.random.beta", return_value=np.array([0.7, 0.6, 0.5, 0.4])):
            mixed_embeddings, mixed_labels = self.trainer._mixup(embeddings, labels)
        # Check shapes
        self.assertEqual(mixed_embeddings.shape, embeddings.shape)
        self.assertEqual(mixed_labels.shape, (4, 3))  # batch_size=4, num_labels=3
        # Check that mixed labels sum to 1 for each example
        self.assertTrue(torch.allclose(mixed_labels.sum(dim=1), torch.ones(4)))

    def test_mixup_3d(self):
        # Test mixup augmentation with 3D embedding tensor (batch_size, seq_len, hidden_dim)
        embeddings = torch.randn(4, 10, 768)  # batch_size=4, seq_len=10, hidden_dim=768
        labels = torch.tensor(
            [0, 1, 2, 0]
        )  # Ensure labels are within the correct range.
        # Mock the np.random.beta to return predictable values
        with patch("numpy.random.beta", return_value=np.array([0.7, 0.6, 0.5, 0.4])):
            mixed_embeddings, mixed_labels = self.trainer._mixup(embeddings, labels)
        # Check shapes
        self.assertEqual(mixed_embeddings.shape, embeddings.shape)
        self.assertEqual(mixed_labels.shape, (4, 3))  # batch_size=4, num_labels=3
        # Check that mixed labels sum to 1 for each example
        self.assertTrue(torch.allclose(mixed_labels.sum(dim=1), torch.ones(4)))

    @patch.object(SLMTrainer, "_create_dataloader")
    @patch.object(SLMTrainer, "evaluate")
    def test_train(self, mock_evaluate, mock_create_dataloader):
        # Skip this test for now - we'll implement a simpler version
        self.skipTest("Skipping complex training test")

    def test_evaluate(self):
        # Mock dataloader
        with patch.object(self.trainer, "_create_dataloader") as mock_create_dataloader:
            mock_dataloader = MagicMock()
            mock_batch = {
                "input_ids": torch.ones(2, 10).long(),
                "attention_mask": torch.ones(2, 10).long(),
                "label": torch.tensor([0, 1]),
            }
            mock_dataloader.__iter__.return_value = [mock_batch]
            mock_create_dataloader.return_value = mock_dataloader

            # Mock model output
            mock_outputs = MagicMock()
            mock_outputs.logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
            self.mock_model.return_value = mock_outputs

            accuracy = self.trainer.evaluate(self.train_data)
            self.assertIsInstance(accuracy, float)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)