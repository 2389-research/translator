#!/usr/bin/env python3
# ABOUTME: Tests for the token counter module.
# ABOUTME: Verifies token counting functionality for various models.

from unittest.mock import patch, MagicMock
from translator.token_counter import TokenCounter
from translator.config import ModelConfig


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


def test_count_tokens_identical_for_o3_and_gpt4():
    """Test that o3 model uses gpt-4 tokenizer."""
    # Sample text
    text = "This is a sample text to ensure consistency in tokenization."

    # Count tokens using different models
    o3_count = TokenCounter.count_tokens(text, "o3")
    gpt4_count = TokenCounter.count_tokens(text, "gpt-4")

    # Since o3 uses gpt-4 encoding, counts should be identical
    assert o3_count == gpt4_count


def test_count_tokens_different_languages():
    """Test token counting for different languages."""
    # Sample texts in different languages
    english_text = "This is English text with about ten tokens."
    spanish_text = "Este es un texto en español con aproximadamente diez tokens."

    # Count tokens
    english_count = TokenCounter.count_tokens(english_text, "gpt-4")
    spanish_count = TokenCounter.count_tokens(spanish_text, "gpt-4")

    # Both should return valid counts
    assert english_count > 0
    assert spanish_count > 0


def test_count_tokens_long_text():
    """Test token counting for longer text samples."""
    # Generate a longer text
    long_text = "This is a longer sample text. " * 50

    # Count tokens
    token_count = TokenCounter.count_tokens(long_text, "gpt-4")

    # Should still work for longer texts
    assert token_count > 200  # Reasonable estimate for 50 repetitions


@patch("tiktoken.encoding_for_model")
def test_count_tokens_fallback(mock_encoding_for_model):
    """Test fallback to cl100k_base when model-specific encoding fails."""
    # Mock encoding_for_model to raise an exception
    mock_encoding_for_model.side_effect = Exception("Model not found")

    # Create a mock encoding for the fallback
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

    # Mock get_encoding to return our mock encoding
    with patch("tiktoken.get_encoding", return_value=mock_encoding):
        # Count tokens with a model that will trigger the fallback
        result = TokenCounter.count_tokens("Test text", "non-existent-model")

        # Should use the fallback encoding and return 5 tokens
        assert result == 5
        mock_encoding.encode.assert_called_once()


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


def test_check_token_limits_at_boundary():
    """Test token limits checking at boundary conditions."""
    # Mock an exact token count
    test_text = "Test text"
    token_count = 1000

    # Mock methods to control the test
    with patch.object(TokenCounter, "count_tokens", return_value=token_count):
        with patch.object(ModelConfig, "get_max_tokens", return_value=2500):
            # At exactly max tokens / 2.5, should be within limits
            within_limits, counted_tokens = TokenCounter.check_token_limits(
                test_text, "test-model"
            )
            assert within_limits is True
            assert counted_tokens == token_count

        with patch.object(ModelConfig, "get_max_tokens", return_value=2499):
            # Just under max tokens / 2.5, should be over limits
            within_limits, counted_tokens = TokenCounter.check_token_limits(
                test_text, "test-model"
            )
            assert within_limits is False
            assert counted_tokens == token_count


def test_check_token_limits_different_models():
    """Test token limits for different models with different max tokens."""
    test_text = "Test text for different models"
    token_count = 10

    # Mock count_tokens to always return a fixed count
    with patch.object(TokenCounter, "count_tokens", return_value=token_count):
        # Test with a small max_tokens model
        with patch.object(ModelConfig, "get_max_tokens", return_value=20):
            within_limits, counted_tokens = TokenCounter.check_token_limits(
                test_text, "small-model"
            )
            assert within_limits is False  # 10 * 2.5 = 25 > 20

        # Test with a large max_tokens model
        with patch.object(ModelConfig, "get_max_tokens", return_value=100):
            within_limits, counted_tokens = TokenCounter.check_token_limits(
                test_text, "large-model"
            )
            assert within_limits is True  # 10 * 2.5 = 25 < 100


def test_check_token_limits_long_content():
    """Test token limits for longer content that exceeds limits."""
    # Create a very long text
    long_text = "This is a very long text. " * 500

    # With a small max token limit
    with patch.object(ModelConfig, "get_max_tokens", return_value=100):
        within_limits, token_count = TokenCounter.check_token_limits(
            long_text, "gpt-3.5-turbo"
        )

        # Should be over limits
        assert within_limits is False
        assert token_count > 40  # Reasonable lower bound for this text


def test_check_token_limits_empty_content():
    """Test token limits for empty content."""
    empty_text = ""

    within_limits, token_count = TokenCounter.check_token_limits(empty_text, "gpt-4")

    # Empty text should have 0 tokens and be within limits
    assert within_limits is True
    assert token_count == 0
