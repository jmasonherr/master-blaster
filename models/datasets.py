import torch

from torch.utils.data import Dataset

from typing import List, Tuple, Dict, Optional, Union, Any, Callable


# DataLoader creation functions
class TextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification tasks.
    This is kept as a class since PyTorch's DataLoader requires a Dataset object.
    """

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }

        return item
