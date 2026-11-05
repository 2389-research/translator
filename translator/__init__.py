#!/usr/bin/env python3
# ABOUTME: Translator package initialization file.
# ABOUTME: Exposes the installed package version.

"""
Translator module for translating text files to different languages using OpenAI or Anthropic models.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("translator")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0"
