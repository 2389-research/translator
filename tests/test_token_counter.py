#!/usr/bin/env python3
# ABOUTME: Tests for the token counter module.
# ABOUTME: Verifies token counting functionality for various models.

import pytest
from translator.token_counter import TokenCounter


def test_count_tokens():
    """Test token counting for different models."""
    # Sample text for testing
    text = "This is a test string for token counting."
    
    # Test with different models
    o3_count = TokenCounter.count_tokens(text, "o3")
    gpt4_count = TokenCounter.count_tokens(text, "gpt-4")
    
    # Assertions
    assert isinstance(o3_count, int)
    assert isinstance(gpt4_count, int)
    assert o3_count > 0
    assert gpt4_count > 0


def test_check_token_limits():
    """Test the token limit checking functionality."""
    # Short text that should be within limits
    short_text = "This is a short test."
    
    # Test with different models
    within_limits, token_count = TokenCounter.check_token_limits(short_text, "o3")
    
    # Assertions
    assert isinstance(within_limits, bool)
    assert isinstance(token_count, int)
    assert within_limits is True
    assert token_count > 0