#!/usr/bin/env python3
# ABOUTME: Language code utilities for handling ISO-639 language codes.
# ABOUTME: Maps language names to standardized codes for file naming.

import re
from typing import Dict

import pycountry


class LanguageHandler:
    """Language code utilities for handling ISO-639 language codes."""

    # Direct lookup for common language names and variations
    LANGUAGE_CODES: Dict[str, str] = {
        "chinese": "zh",
        "mandarin": "zh",
        "spanish": "es",
        "español": "es",
        "english": "en",
        "hindi": "hi",
        "arabic": "ar",
        "portuguese": "pt",
        "brazilian": "pt",
        "bengali": "bn",
        "russian": "ru",
        "japanese": "ja",
        "punjabi": "pa",
        "german": "de",
        "deutsch": "de",
        "javanese": "jv",
        "korean": "ko",
        "french": "fr",
        "français": "fr",
        "turkish": "tr",
        "vietnamese": "vi",
        "thai": "th",
        "italian": "it",
        "italiano": "it",
        "persian": "fa",
        "farsi": "fa",
        "polish": "pl",
        "polski": "pl",
        "romanian": "ro",
        "dutch": "nl",
        "greek": "el",
        "czech": "cs",
        "swedish": "sv",
        "hebrew": "he",
        "danish": "da",
        "finnish": "fi",
        "hungarian": "hu",
        "norwegian": "no",
    }

    @classmethod
    def get_language_code(cls, language_name: str) -> str:
        """Convert a language name to ISO 639-1 two-letter code.

        Args:
            language_name: The name of the language to convert

        Returns:
            The ISO 639-1 two-letter code for the language
        """
        # Normalize input: lowercase and remove any non-alphanumeric characters
        language_name_normalized = re.sub(
            r"[^a-zA-Z0-9]", " ", language_name.lower()
        ).strip()

        # First try direct mapping
        if language_name_normalized in cls.LANGUAGE_CODES:
            return cls.LANGUAGE_CODES[language_name_normalized]

        # Try with pycountry
        try:
            # Try to find by name
            lang = pycountry.languages.get(name=language_name_normalized.title())
            if lang and hasattr(lang, "alpha_2"):
                return lang.alpha_2

            # Try to find by partial name match
            for lang in pycountry.languages:
                if (
                    hasattr(lang, "name")
                    and language_name_normalized in lang.name.lower()
                    and hasattr(lang, "alpha_2")
                ):
                    return lang.alpha_2
        except (AttributeError, KeyError):
            pass

        # Fall back to using the first two letters of the language name
        return language_name_normalized[:2]
