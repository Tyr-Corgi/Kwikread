#!/usr/bin/env python3
"""
Evaluation Metrics for Handwriting OCR

Implements Character Error Rate (CER) and Word Error Rate (WER)
for evaluating TrOCR model performance.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = total characters in reference

    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings

    Returns:
        CER as a float between 0 and 1 (or higher if very poor)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    total_distance = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        # Normalize strings
        pred = pred.strip()
        ref = ref.strip()

        total_distance += levenshtein_distance(pred, ref)
        total_chars += len(ref)

    if total_chars == 0:
        return 0.0

    return total_distance / total_chars


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (S + D + I) / N
    where the operations are on words, not characters.

    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings

    Returns:
        WER as a float between 0 and 1 (or higher if very poor)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    total_distance = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        # Tokenize into words
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()

        # Use edit distance on word lists
        distance = word_levenshtein_distance(pred_words, ref_words)

        total_distance += distance
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0

    return total_distance / total_words


def word_levenshtein_distance(s1: List[str], s2: List[str]) -> int:
    """
    Calculate Levenshtein distance between two word lists.
    """
    if len(s1) < len(s2):
        return word_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, w1 in enumerate(s1):
        current_row = [i + 1]
        for j, w2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1 != w2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match accuracy.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip() == ref.strip()
    )

    return correct / len(predictions)


def evaluate_model(
    predictions: List[str],
    references: List[str],
    detailed: bool = False
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        predictions: List of predicted strings
        references: List of reference strings
        detailed: Include per-sample metrics

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "cer": calculate_cer(predictions, references),
        "wer": calculate_wer(predictions, references),
        "accuracy": calculate_accuracy(predictions, references),
        "num_samples": len(predictions)
    }

    if detailed:
        sample_metrics = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred = pred.strip()
            ref = ref.strip()

            sample_cer = levenshtein_distance(pred, ref) / max(len(ref), 1)
            pred_words = pred.split()
            ref_words = ref.split()
            sample_wer = word_levenshtein_distance(pred_words, ref_words) / max(len(ref_words), 1)

            sample_metrics.append({
                "index": i,
                "prediction": pred,
                "reference": ref,
                "cer": sample_cer,
                "wer": sample_wer,
                "exact_match": pred == ref
            })

        metrics["samples"] = sample_metrics

    return metrics


def print_evaluation_report(metrics: Dict, top_errors: int = 10):
    """
    Print a formatted evaluation report.

    Args:
        metrics: Dictionary from evaluate_model()
        top_errors: Number of worst predictions to show
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"Word Error Rate (WER): {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"Exact Match Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    if "samples" in metrics and top_errors > 0:
        print("\n" + "-" * 60)
        print(f"Top {top_errors} Worst Predictions (by CER):")
        print("-" * 60)

        # Sort by CER descending
        sorted_samples = sorted(
            metrics["samples"],
            key=lambda x: x["cer"],
            reverse=True
        )

        for sample in sorted_samples[:top_errors]:
            print(f"\n[{sample['index']}] CER: {sample['cer']:.4f}")
            print(f"  Reference:  '{sample['reference']}'")
            print(f"  Prediction: '{sample['prediction']}'")

    print("\n" + "=" * 60)


# Try to use jiwer if available (more optimized implementation)
try:
    import jiwer

    def calculate_cer_jiwer(predictions: List[str], references: List[str]) -> float:
        """Calculate CER using jiwer library."""
        # Use jiwer.cer for character error rate (newer API)
        try:
            return jiwer.cer(references, predictions)
        except Exception:
            # Fallback for older jiwer API
            transform = jiwer.Compose([
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfChars()
            ])
            return jiwer.wer(
                references,
                predictions,
                reference_transform=transform,
                hypothesis_transform=transform
            )

    def calculate_wer_jiwer(predictions: List[str], references: List[str]) -> float:
        """Calculate WER using jiwer library."""
        return jiwer.wer(references, predictions)

    # Override with jiwer implementations
    calculate_cer = calculate_cer_jiwer
    calculate_wer = calculate_wer_jiwer
    print("Using jiwer for metrics calculation")

except ImportError:
    pass  # Use pure Python implementation


if __name__ == "__main__":
    # Test metrics
    predictions = [
        "hello world",
        "the quick brown fox",
        "handwriting recognition",
        "test"
    ]

    references = [
        "hello world",
        "the quik brown fox",  # Intentional typo
        "handwriting recgnition",  # Intentional typo
        "testing"
    ]

    print("Testing metrics calculation...")
    metrics = evaluate_model(predictions, references, detailed=True)
    print_evaluation_report(metrics)
