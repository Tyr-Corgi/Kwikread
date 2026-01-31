"""
Grocery Item Corrector - Fuzzy matches OCR text to known grocery items.
Uses edit distance to correct common OCR errors.
"""

from difflib import SequenceMatcher
from typing import Optional, Tuple
import re

# Common grocery items (expandable)
GROCERY_ITEMS = [
    # Dairy
    "milk", "oat milk", "rice milk", "almond milk", "cheese", "cheese sticks",
    "butter", "yogurt", "cream cheese", "sour cream", "eggs",
    # Bread/Grains
    "bread", "rolls", "rhodes rolls", "tortilla", "tortillas", "bagels",
    "cereal", "oats", "oatmeal", "rice", "pasta", "noodles",
    # Produce
    "apples", "apple sauce", "applesauce", "bananas", "strawberry", "strawberries",
    "potato", "potatoes", "lettuce", "tomato", "tomatoes", "onion", "onions",
    "carrot", "carrots", "broccoli", "spinach",
    # Frozen
    "ice cream", "french fries", "frozen pizza", "frozen vegetables",
    # Snacks
    "trail mix", "protein bars", "protein bar", "chips", "crackers", "cookies",
    "cookie dough", "granola", "granola bars", "nuts",
    # Meat
    "chicken", "beef", "pork", "bacon", "ham", "turkey", "cold cut", "coldcut",
    # Beverages
    "juice", "orange juice", "coffee", "tea", "soda", "water",
    # Condiments
    "ketchup", "mustard", "mayo", "mayonnaise", "salsa",
    # Flavors (ice cream)
    "vanilla", "chocolate", "mint", "strawberry", "cookie dough",
    # Brands (commonly OCR'd)
    "pillsbury", "pillsbury cookie dough",
]

# Word-level corrections for common OCR errors
WORD_CORRECTIONS = {
    "philbury": "pillsbury",
    "page": "dough",
    "erie's": "fries",
    "eries": "fries",
    "unit": "mix",
    "out": "oat",
    "protien": "protein",
    "bar's": "bars",
}

# Build lowercase lookup
GROCERY_SET = {item.lower() for item in GROCERY_ITEMS}


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(text: str, threshold: float = 0.6) -> Tuple[Optional[str], float]:
    """
    Find the best matching grocery item for the given text.

    Args:
        text: OCR text to match
        threshold: Minimum similarity score (0-1)

    Returns:
        Tuple of (best_match, score) or (None, 0) if no match found
    """
    text_clean = text.lower().strip().rstrip('.')

    # Exact match
    if text_clean in GROCERY_SET:
        return text_clean, 1.0

    best_match = None
    best_score = 0.0

    for item in GROCERY_ITEMS:
        score = similarity(text_clean, item)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = item

    return best_match, best_score


def correct_grocery_text(text: str, threshold: float = 0.65) -> Tuple[str, bool, float]:
    """
    Attempt to correct OCR text to a known grocery item.

    Args:
        text: OCR text to correct
        threshold: Minimum similarity for correction

    Returns:
        Tuple of (corrected_text, was_corrected, confidence)
    """
    original = text.strip()

    # Remove trailing periods/punctuation for matching
    text_clean = re.sub(r'[.\s]+$', '', original)

    # First try exact/full match
    match, score = find_best_match(text_clean, threshold)

    if match and score >= threshold:
        # Preserve original capitalization style
        if text_clean and text_clean[0].isupper():
            corrected = match.title()
        else:
            corrected = match
        return corrected, score < 1.0, score

    # Apply word-level corrections first
    words = text_clean.split()
    corrected_words = []
    any_word_corrected = False

    for word in words:
        word_lower = word.lower().rstrip(".'")
        if word_lower in WORD_CORRECTIONS:
            corrected = WORD_CORRECTIONS[word_lower]
            if word[0].isupper():
                corrected = corrected.title()
            corrected_words.append(corrected)
            any_word_corrected = True
        else:
            corrected_words.append(word)

    if any_word_corrected:
        text_clean = ' '.join(corrected_words)
        # Try full match again with corrected text
        match, score = find_best_match(text_clean, threshold)
        if match and score >= threshold:
            if text_clean and text_clean[0].isupper():
                return match.title(), True, score
            return match, True, score
        return text_clean, True, 0.8  # Word corrections applied

    # Try matching individual words for compound phrases
    if len(words) >= 2:
        corrected_words = []
        any_corrected = False
        total_score = 0

        for word in words:
            word_match, word_score = find_best_match(word, threshold)
            if word_match and word_score >= threshold:
                if word[0].isupper():
                    corrected_words.append(word_match.title())
                else:
                    corrected_words.append(word_match)
                any_corrected = any_corrected or (word_score < 1.0)
                total_score += word_score
            else:
                corrected_words.append(word)
                total_score += 0.5  # Partial credit for unknown words

        if any_corrected:
            avg_score = total_score / len(words)
            return ' '.join(corrected_words), True, avg_score

    return original, False, 0.0


def correct_line(text: str, threshold: float = 0.65) -> str:
    """Simple wrapper to correct a line of text."""
    corrected, was_corrected, score = correct_grocery_text(text, threshold)
    return corrected


# Test
if __name__ == "__main__":
    test_cases = [
        "French Erie's",
        "Trail unit",
        "out milk",
        "Protien Bar's",
        "evamilla",
        "rice",
        "Philbury Cookie page",
        "Strawberry",
        "Ice cream",
        "Oats - Cold Cut",
    ]

    print("Grocery Corrector Test:")
    print("-" * 50)
    for text in test_cases:
        corrected, was_corrected, score = correct_grocery_text(text)
        status = "→" if was_corrected else "="
        print(f"  {text:25s} {status} {corrected:20s} ({score:.2f})")
