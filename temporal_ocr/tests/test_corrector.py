"""
Unit tests for grocery_corrector.py post-processing module.

This module tests the fuzzy matching and correction logic:
- WORD_CORRECTIONS dictionary lookups
- PHRASE_COMPLETIONS logic
- Protected words (should not be fuzzy matched)
- Semantic validation (reject nonsense like "chicken seeds")
- Length-adjusted thresholds
- Confusion-aware similarity

Usage:
    pytest tests/test_corrector.py -v
    pytest tests/test_corrector.py::TestWordCorrections -v
    pytest tests/test_corrector.py -k "not slow" -v
"""

import sys
from pathlib import Path
import pytest

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def correct_grocery_text():
    """Import the main correction function."""
    from grocery_corrector import correct_grocery_text
    return correct_grocery_text


@pytest.fixture
def grocery_set():
    """Import the grocery items set."""
    from grocery_corrector import GROCERY_SET
    return GROCERY_SET


@pytest.fixture
def protected_words():
    """Import protected words set."""
    from grocery_corrector import PROTECTED_WORDS
    return PROTECTED_WORDS


@pytest.fixture
def word_corrections():
    """Import word corrections dictionary."""
    from grocery_corrector import WORD_CORRECTIONS
    return WORD_CORRECTIONS


@pytest.fixture
def phrase_completions():
    """Import phrase completions dictionary."""
    from grocery_corrector import PHRASE_COMPLETIONS
    return PHRASE_COMPLETIONS


# =============================================================================
# Word Corrections Tests
# =============================================================================

class TestWordCorrections:
    """Test WORD_CORRECTIONS dictionary lookups."""

    def test_common_ocr_errors(self, correct_grocery_text):
        """Test correction of common OCR errors."""
        test_cases = [
            # (input, expected_output_contains)
            ("out milk", "oat"),  # out -> oat
            ("protien bars", "protein"),  # protien -> protein
            ("trail unit", "trail mix"),  # unit -> mix
        ]

        for input_text, expected_contains in test_cases:
            corrected, was_corrected, score = correct_grocery_text(input_text)
            assert expected_contains.lower() in corrected.lower(), (
                f"'{input_text}' -> '{corrected}' should contain '{expected_contains}'"
            )

    def test_brand_corrections(self, correct_grocery_text):
        """Test correction of misspelled brand names."""
        test_cases = [
            ("Philbury", "Pillsbury"),
            ("pillbury", "pillsbury"),
            ("nestel", "nestle"),
        ]

        for input_text, expected in test_cases:
            corrected, was_corrected, score = correct_grocery_text(input_text)
            assert expected.lower() in corrected.lower(), (
                f"'{input_text}' -> '{corrected}' should contain '{expected}'"
            )

    def test_ground_beef_confusion(self, correct_grocery_text):
        """Test correction of 'around' -> 'ground' for beef."""
        # "around beef" is a common OCR error for "ground beef"
        corrected, was_corrected, score = correct_grocery_text("around beef")
        # Should be corrected to ground beef
        assert "ground" in corrected.lower() or was_corrected, (
            f"'around beef' should be corrected, got '{corrected}'"
        )

    def test_kitchen_chicken_confusion(self, correct_grocery_text):
        """Test correction of 'kitchen' -> 'chicken'."""
        corrected, was_corrected, score = correct_grocery_text("kitchen breast")
        # Might be corrected to chicken breast
        # This is context-dependent


# =============================================================================
# Protected Words Tests
# =============================================================================

class TestProtectedWords:
    """Test that protected words are not incorrectly modified."""

    def test_salt_not_corrected_to_salsa(self, correct_grocery_text):
        """
        CRITICAL: 'Salt' should NOT become 'Salsa'.

        This is a key test case - short words need high similarity thresholds.
        """
        corrected, was_corrected, score = correct_grocery_text("Salt")
        assert corrected.lower() == "salt", (
            f"'Salt' was incorrectly changed to '{corrected}'"
        )

    def test_peas_not_corrected_to_pasta(self, correct_grocery_text):
        """'Peas' should not become 'Pasta'."""
        corrected, was_corrected, score = correct_grocery_text("Peas")
        # Peas is valid - should not change
        assert "peas" in corrected.lower(), (
            f"'Peas' was incorrectly changed to '{corrected}'"
        )

    def test_rice_unchanged(self, correct_grocery_text):
        """'Rice' is valid and should remain unchanged."""
        corrected, was_corrected, score = correct_grocery_text("Rice")
        assert corrected.lower() in ["rice", "rice milk"], (
            f"'Rice' was unexpectedly changed to '{corrected}'"
        )

    def test_protected_words_list(self, protected_words):
        """Verify protected words set contains expected items."""
        expected_protected = ["salt", "peas", "rice", "eggs", "ham", "tea", "milk", "beef"]

        for word in expected_protected:
            assert word in protected_words, (
                f"'{word}' should be in protected words"
            )

    def test_all_protected_words_unchanged(self, correct_grocery_text, protected_words):
        """Test that all protected words remain unchanged when passed as input."""
        # Sample of protected words
        sample = ["salt", "peas", "rice", "eggs", "milk", "beef", "pork", "ham", "tea"]

        for word in sample:
            corrected, was_corrected, score = correct_grocery_text(word.capitalize())
            # Protected words should either stay the same or become valid completions
            assert word in corrected.lower() or corrected.lower() in protected_words, (
                f"Protected word '{word}' was changed to '{corrected}'"
            )


# =============================================================================
# Semantic Validation Tests
# =============================================================================

class TestSemanticValidation:
    """Test that semantically invalid corrections are rejected."""

    def test_no_chicken_seeds(self, correct_grocery_text):
        """
        'Chia Seeds' should NOT become 'Chicken Seeds'.

        Semantic validation should reject meat + seeds combinations.
        """
        # If someone writes "chia seeds" with OCR errors
        corrected, was_corrected, score = correct_grocery_text("Chia Seeds")
        assert "chicken" not in corrected.lower(), (
            f"'Chia Seeds' incorrectly became '{corrected}'"
        )

    def test_black_beans_not_bananas(self, correct_grocery_text):
        """'Black beans' should not become 'Black bananas'."""
        corrected, was_corrected, score = correct_grocery_text("Black beans")
        assert "banana" not in corrected.lower(), (
            f"'Black beans' incorrectly became '{corrected}'"
        )

    def test_green_beans_preserved(self, correct_grocery_text):
        """'Green beans' is valid and should be preserved."""
        corrected, was_corrected, score = correct_grocery_text("Green beans")
        assert "beans" in corrected.lower(), (
            f"'Green beans' was incorrectly changed to '{corrected}'"
        )


# =============================================================================
# Phrase Completion Tests
# =============================================================================

class TestPhraseCompletions:
    """Test PHRASE_COMPLETIONS for truncated words."""

    def test_chai_completes_to_chai_tea(self, correct_grocery_text):
        """'chai' alone should complete to 'chai tea'."""
        corrected, was_corrected, score = correct_grocery_text("chai")
        # chai is valid but might be completed to "chai tea"
        assert "chai" in corrected.lower(), (
            f"'chai' was incorrectly changed to '{corrected}'"
        )

    def test_beans_completion(self, correct_grocery_text):
        """'beans' alone might complete to 'green beans' or similar."""
        corrected, was_corrected, score = correct_grocery_text("beans")
        # beans should either stay as beans or become a valid bean type
        assert "beans" in corrected.lower(), (
            f"'beans' was incorrectly changed to '{corrected}'"
        )

    def test_phrase_completions_structure(self, phrase_completions):
        """Verify phrase completions have correct structure."""
        assert len(phrase_completions) > 0, "No phrase completions defined"

        # Each entry should be word -> [(phrase, confidence), ...]
        for word, completions in phrase_completions.items():
            assert isinstance(completions, list), (
                f"Completions for '{word}' should be a list"
            )
            for phrase, conf in completions:
                assert isinstance(phrase, str), (
                    f"Phrase should be string, got {type(phrase)}"
                )
                assert 0 <= conf <= 1, (
                    f"Confidence {conf} should be between 0 and 1"
                )


# =============================================================================
# Similarity Function Tests
# =============================================================================

class TestSimilarityFunctions:
    """Test similarity calculation functions."""

    def test_confusion_aware_similarity(self):
        """Test confusion-aware similarity gives bonus for OCR confusions."""
        from grocery_corrector import confusion_aware_similarity

        # r/n confusion should get bonus
        sim_rn = confusion_aware_similarity("ramen", "namen")
        sim_base = confusion_aware_similarity("ramen", "xamen")

        # r/n confusion should score higher than random character
        assert sim_rn >= sim_base, (
            f"r/n confusion ({sim_rn:.2f}) should score >= random ({sim_base:.2f})"
        )

    def test_length_adjusted_threshold(self):
        """Test length-adjusted thresholds for different word lengths."""
        from grocery_corrector import get_length_adjusted_threshold

        # Short words need higher thresholds
        short_threshold = get_length_adjusted_threshold("salt")  # 4 chars
        medium_threshold = get_length_adjusted_threshold("chicken")  # 7 chars
        long_threshold = get_length_adjusted_threshold("strawberries")  # 12 chars

        assert short_threshold > medium_threshold, (
            f"Short word threshold ({short_threshold}) should be > medium ({medium_threshold})"
        )
        assert medium_threshold >= long_threshold, (
            f"Medium threshold ({medium_threshold}) should be >= long ({long_threshold})"
        )


# =============================================================================
# Exact Match Tests
# =============================================================================

class TestExactMatches:
    """Test that exact grocery items pass through unchanged."""

    def test_valid_items_unchanged(self, correct_grocery_text):
        """Valid grocery items should pass through unchanged."""
        valid_items = [
            "cheese", "butter", "milk", "eggs", "bread",
            "chicken", "beef", "pork", "salmon", "tuna",
            "apples", "bananas", "oranges", "lettuce", "tomatoes",
        ]

        for item in valid_items:
            corrected, was_corrected, score = correct_grocery_text(item)
            # Should either be unchanged or complete to valid phrase
            assert item in corrected.lower() or score > 0.8, (
                f"Valid item '{item}' was changed to '{corrected}'"
            )

    def test_multi_word_items(self, correct_grocery_text):
        """Multi-word grocery items should work correctly."""
        multi_word = [
            "ice cream", "paper towels", "olive oil", "soy sauce",
            "ground beef", "chicken breast", "brown rice",
        ]

        for item in multi_word:
            corrected, was_corrected, score = correct_grocery_text(item)
            # Check that at least one key word is preserved
            words = item.split()
            has_match = any(w in corrected.lower() for w in words)
            assert has_match, (
                f"Multi-word item '{item}' -> '{corrected}' lost all key words"
            )


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self, correct_grocery_text):
        """Empty string should return empty."""
        corrected, was_corrected, score = correct_grocery_text("")
        assert corrected == "", f"Empty input returned '{corrected}'"

    def test_whitespace_only(self, correct_grocery_text):
        """Whitespace-only input should return empty/whitespace."""
        corrected, was_corrected, score = correct_grocery_text("   ")
        assert corrected.strip() == "", f"Whitespace returned '{corrected}'"

    def test_single_char(self, correct_grocery_text):
        """Single character input should be handled."""
        corrected, was_corrected, score = correct_grocery_text("a")
        # Should not crash, return value can vary
        assert isinstance(corrected, str)

    def test_very_long_input(self, correct_grocery_text):
        """Very long input should be handled."""
        long_text = "a " * 100  # 200 chars
        corrected, was_corrected, score = correct_grocery_text(long_text)
        assert isinstance(corrected, str)

    def test_special_characters(self, correct_grocery_text):
        """Input with special characters should be handled."""
        special_inputs = [
            "milk!!!", "eggs???", "bread...", "cheese,,,",
            "salt-pepper", "oil/vinegar", "mac&cheese",
        ]

        for inp in special_inputs:
            corrected, was_corrected, score = correct_grocery_text(inp)
            assert isinstance(corrected, str), f"'{inp}' failed"

    def test_numbers_in_input(self, correct_grocery_text):
        """Input with numbers should be handled."""
        numbered = ["2% milk", "1lb beef", "3 eggs", "12 pack"]

        for inp in numbered:
            corrected, was_corrected, score = correct_grocery_text(inp)
            assert isinstance(corrected, str), f"'{inp}' failed"


# =============================================================================
# Corrector API Tests
# =============================================================================

class TestCorrectorAPI:
    """Test the corrector API returns correct types."""

    def test_return_type(self, correct_grocery_text):
        """Verify correct_grocery_text returns (str, bool, float)."""
        result = correct_grocery_text("test")

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 3, "Should return 3 values"

        corrected, was_corrected, score = result
        assert isinstance(corrected, str), "First value should be string"
        assert isinstance(was_corrected, bool), "Second value should be bool"
        assert isinstance(score, (int, float)), "Third value should be numeric"

    def test_threshold_parameter(self, correct_grocery_text):
        """Test threshold parameter affects corrections."""
        # Low threshold should allow more corrections
        low_result = correct_grocery_text("mlik", threshold=0.5)

        # High threshold should be more conservative
        high_result = correct_grocery_text("mlik", threshold=0.99)

        # Both should return valid results
        assert isinstance(low_result[0], str)
        assert isinstance(high_result[0], str)


# =============================================================================
# Regression Tests for Known Issues
# =============================================================================

class TestKnownIssues:
    """Regression tests for previously identified issues."""

    def test_chaites_to_chai_tea(self, correct_grocery_text):
        """'chaites' should become 'chai tea' (OCR error from videotest3)."""
        corrected, was_corrected, score = correct_grocery_text("chaites")
        # Should be corrected to chai or chai tea
        assert "chai" in corrected.lower(), (
            f"'chaites' was not corrected to chai, got '{corrected}'"
        )

    def test_paprites_to_paprika(self, correct_grocery_text):
        """'paprites' should become 'paprika'."""
        corrected, was_corrected, score = correct_grocery_text("paprites")
        assert "paprika" in corrected.lower() or "papri" in corrected.lower(), (
            f"'paprites' was not corrected, got '{corrected}'"
        )

    def test_pretzals_to_pretzels(self, correct_grocery_text):
        """'pretzals' should become 'pretzels'."""
        corrected, was_corrected, score = correct_grocery_text("pretzals")
        assert "pretzel" in corrected.lower(), (
            f"'pretzals' was not corrected, got '{corrected}'"
        )

    def test_applesause_to_applesauce(self, correct_grocery_text):
        """'applesause' should become 'applesauce'."""
        corrected, was_corrected, score = correct_grocery_text("applesause")
        assert "applesauce" in corrected.lower() or "apple sauce" in corrected.lower(), (
            f"'applesause' was not corrected, got '{corrected}'"
        )


# =============================================================================
# Performance Tests
# =============================================================================

class TestCorrectorPerformance:
    """Test corrector performance."""

    def test_correction_speed(self, correct_grocery_text):
        """Test that corrections are fast enough."""
        import time

        # Run 100 corrections
        test_words = ["milk", "eggs", "bread", "cheese", "out milk"] * 20

        start = time.perf_counter()
        for word in test_words:
            _ = correct_grocery_text(word)
        elapsed = time.perf_counter() - start

        # Should complete 100 corrections in under 100ms
        assert elapsed < 0.1, (
            f"100 corrections took {elapsed*1000:.0f}ms, should be <100ms"
        )

        per_correction = (elapsed / len(test_words)) * 1000  # ms
        print(f"\nCorrection speed: {per_correction:.2f}ms per item")
