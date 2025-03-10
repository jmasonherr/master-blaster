import anthropic
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from Levenshtein import distance


@dataclass
class ClassificationResult:
    """Data class to hold classification results with fuzzy label matching"""

    text: str
    raw_label: str
    matched_label: str
    confidence: float
    confidence_score: int

    def __init__(
        self,
        text: str,
        raw_label: str,
        confidence: float,
        valid_labels: List[str],
    ):
        self.text = text
        self.raw_label = raw_label
        self.confidence = confidence
        self.matched_label = self._find_best_label_match(raw_label, valid_labels)
        self.confidence_score = self._calculate_confidence_score()

    def _find_best_label_match(self, label: str, valid_labels: List[str]) -> str:
        """Find the closest matching label using Levenshtein distance"""
        # Clean up the label
        clean_label = label.lower().strip()

        # Check for exact match
        for valid_label in valid_labels:
            if valid_label.lower() == clean_label:
                return valid_label

        # Use Levenshtein distance for fuzzy matching
        best_label = None
        best_distance = float("inf")

        for valid_label in valid_labels:
            dist = distance(valid_label.lower(), clean_label)
            if dist < best_distance:
                best_distance = dist
                best_label = valid_label

        # If the best distance is too high, return the original
        if best_distance > len(best_label) / 2:
            return label

        return best_label

    def _calculate_confidence_score(self) -> int:
        """Calculate confidence score on scale of 1-10 based on various factors"""
        # Start with base score derived from confidence value if available
        base_score = int(self.confidence * 10) if self.confidence else 5

        # Adjust based on label matching
        if self.raw_label.lower() == self.matched_label.lower():
            # Exact match - no penalty
            label_factor = 0
        else:
            # Apply penalty for fuzzy matching
            label_factor = -2

        # Final score calculation with bounds checking
        final_score = max(1, min(10, base_score + label_factor))
        return final_score
