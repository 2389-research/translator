#!/usr/bin/env python3
# ABOUTME: Tests for the language handler module.
# ABOUTME: Verifies language code detection functionality.

import pytest
from translator.language import LanguageHandler


def test_get_language_code_common_languages():
    """Test language code detection for common languages."""
    assert LanguageHandler.get_language_code("English") == "en"
    assert LanguageHandler.get_language_code("Spanish") == "es"
    assert LanguageHandler.get_language_code("French") == "fr"
    assert LanguageHandler.get_language_code("German") == "de"
    assert LanguageHandler.get_language_code("Japanese") == "ja"
    assert LanguageHandler.get_language_code("Chinese") == "zh"


def test_get_language_code_case_insensitivity():
    """Test case insensitivity in language detection."""
    assert LanguageHandler.get_language_code("english") == "en"
    assert LanguageHandler.get_language_code("SPANISH") == "es"


def test_get_language_code_complex_names():
    """Test language code detection for complex language names."""
    assert LanguageHandler.get_language_code("Modern Greek") == "el"
    assert LanguageHandler.get_language_code("Brazilian") == "pt"


def test_get_language_code_fallback():
    """Test fallback behavior for unknown languages."""
    unknown_lang = "NonExistentLanguage"
    result = LanguageHandler.get_language_code(unknown_lang)
    assert isinstance(result, str)
    assert len(result) == 2  # Should be a 2-letter code