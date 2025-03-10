#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import json
import torch
import logging

from typing import List, Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_row_as_text(row: pd.Series, exclude_empty: bool = True) -> str:
    """
    Format a pandas row as a text string with field names.

    Args:
        row: A pandas Series representing a row
        exclude_empty: Whether to exclude empty fields

    Returns:
        Formatted text string
    """
    parts = []
    for column, value in row.items():
        # Skip empty values if requested
        if exclude_empty and (pd.isna(value) or value == ""):
            continue
        # Format the field as "column_name: value"
        parts.append(f"{column}: {value}")

    # Join all parts with commas
    return ", ".join(parts)

