#!/usr/bin/env python3
# ABOUTME: Tests for the translator module with provider architecture.
# ABOUTME: Verifies translation functionality and provider interactions.

import pytest
from unittest.mock import patch, MagicMock
import openai
from translator.translator import Translator


@pytest.fixture
def openai_client():
    """Create a mock OpenAI client for testing."""
    mock_client = MagicMock(spec=openai.OpenAI)
    return mock_client


@pytest.fixture
def translator_instance(openai_client):
    """Create a translator instance with mock client for testing."""
    return Translator(openai_client=openai_client)


def test_init(openai_client):
    """Test translator initialization."""
    translator = Translator(openai_client=openai_client)

    assert translator.openai_client == openai_client
    assert translator.anthropic_client is None
    assert translator.translation_context == ""
    assert isinstance(translator.translation_log, dict)
    assert "translation" in translator.translation_log
    assert "editing" in translator.translation_log
    assert "critique" in translator.translation_log
    assert "feedback" in translator.translation_log
    assert "frontmatter" in translator.translation_log
    assert "all_critiques" in translator.translation_log


@patch('translator.translator.ProviderFactory.create_provider')
def test_translate_text_without_context(mock_provider_factory, translator_instance):
    """Test the translate_text method without context."""
    # Mock the provider and its response
    mock_provider = MagicMock()
    mock_provider.translate_text.return_value = (
        "Texto traducido al español.",
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        None
    )
    mock_provider_factory.return_value = mock_provider

    # Call the method
    text = "Text to be translated."
    target_language = "Spanish"
    model = "gpt-4"

    translated_text, usage, error_msg = translator_instance.translate_text(
        text, target_language, model
    )

    # Verify results
    assert translated_text == "Texto traducido al español."
    assert error_msg is None
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30

    # Verify the provider factory was called correctly
    mock_provider_factory.assert_called_once_with(
        model, openai_client=translator_instance.openai_client, anthropic_client=translator_instance.anthropic_client
    )

    # Verify the provider was called correctly
    mock_provider.translate_text.assert_called_once()

    # Verify translation log was updated
    assert "model" in translator_instance.translation_log["translation"]
    assert translator_instance.translation_log["translation"]["target_language"] == "Spanish"


@patch('translator.translator.ProviderFactory.create_provider')
def test_translate_text_error(mock_provider_factory, translator_instance):
    """Test error handling in translate_text method."""
    # Mock the provider factory to raise an exception
    mock_provider_factory.side_effect = Exception("Provider Error")

    # Call the method and check for error message
    translated_text, usage, error_msg = translator_instance.translate_text(
        "Text to translate", "Spanish", "gpt-4"
    )

    # Verify error handling
    assert translated_text is None
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert error_msg is not None
    assert "Provider Error" in error_msg


@patch('translator.translator.ProviderFactory.create_provider')
def test_edit_translation(mock_provider_factory, translator_instance):
    """Test the edit_translation method."""
    # Mock the provider and its response
    mock_provider = MagicMock()
    mock_provider.translate_text.return_value = (
        "Texto mejorado en español.",
        {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        None
    )
    mock_provider_factory.return_value = mock_provider

    # Call the method
    translated_text = "Texto traducido al español."
    original_text = "Text to be translated."
    target_language = "Spanish"
    model = "gpt-4"

    edited_text, usage, error_msg = translator_instance.edit_translation(
        translated_text, original_text, target_language, model
    )

    # Verify results
    assert edited_text == "Texto mejorado en español."
    assert usage["prompt_tokens"] == 15
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 40
    assert error_msg is None

    # Verify editing log was updated
    assert "model" in translator_instance.translation_log["editing"]
    assert translator_instance.translation_log["editing"]["target_language"] == "Spanish"


@patch('translator.translator.ProviderFactory.create_provider')
def test_critique_translation(mock_provider_factory, translator_instance):
    """Test the critique_translation method."""
    # Mock the provider and its response
    mock_provider = MagicMock()
    critique = "La traducción necesita mejoras en el tono y la naturalidad."
    mock_provider.translate_text.return_value = (
        critique,
        {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        None
    )
    mock_provider_factory.return_value = mock_provider

    # Call the method
    translated_text = "Texto traducido al español."
    original_text = "Text to be translated."
    target_language = "Spanish"
    model = "gpt-4"

    result_text, usage, feedback, error_msg = translator_instance.critique_translation(
        translated_text, original_text, target_language, model
    )

    # Verify results
    assert result_text == translated_text  # Critique doesn't change the text
    assert feedback == critique
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 30
    assert usage["total_tokens"] == 50
    assert error_msg is None

    # Verify critique log was updated
    assert "model" in translator_instance.translation_log["critique"]
    assert translator_instance.translation_log["critique"]["target_language"] == "Spanish"


@patch('translator.translator.ProviderFactory.create_provider')
def test_apply_critique_feedback(mock_provider_factory, translator_instance):
    """Test the apply_critique_feedback method."""
    # Mock the provider and its response
    mock_provider = MagicMock()
    improved_text = "Texto mejorado basado en la crítica."
    mock_provider.translate_text.return_value = (
        improved_text,
        {"prompt_tokens": 25, "completion_tokens": 35, "total_tokens": 60},
        None
    )
    mock_provider_factory.return_value = mock_provider

    # Call the method
    translated_text = "Texto traducido al español."
    original_text = "Text to be translated."
    critique = "La traducción necesita mejoras en fluidez."
    target_language = "Spanish"
    model = "gpt-4"

    result, usage, error_msg = translator_instance.apply_critique_feedback(
        translated_text, original_text, critique, target_language, model
    )

    # Verify results
    assert result == improved_text
    assert usage["prompt_tokens"] == 25
    assert usage["completion_tokens"] == 35
    assert usage["total_tokens"] == 60
    assert error_msg is None

    # Verify feedback log was updated
    assert "model" in translator_instance.translation_log["feedback"]
    assert translator_instance.translation_log["feedback"]["target_language"] == "Spanish"


@patch('translator.translator.ProviderFactory.create_provider')
def test_translate_frontmatter(mock_provider_factory, translator_instance):
    """Test the translate_frontmatter method."""
    # Mock the provider and its response
    mock_provider = MagicMock()
    response_text = """title: Título Traducido

description: Descripción traducida"""
    mock_provider.translate_text.return_value = (
        response_text,
        {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70},
        None
    )
    mock_provider_factory.return_value = mock_provider

    # Call the method
    frontmatter_data = {
        "title": "Original Title",
        "description": "Original Description",
        "date": "2023-01-01",
        "author": "Author Name",
    }
    fields = ["title", "description"]

    result, usage, error_msg = translator_instance.translate_frontmatter(
        frontmatter_data, fields, "Spanish", "gpt-4"
    )

    # Verify results
    assert result["title"] == "Título Traducido"
    assert result["description"] == "Descripción traducida"
    assert result["date"] == "2023-01-01"  # Unchanged
    assert result["author"] == "Author Name"  # Unchanged

    assert usage["prompt_tokens"] == 30
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 70
    assert error_msg is None

    # Verify frontmatter log was updated
    assert "model" in translator_instance.translation_log["frontmatter"]
    assert translator_instance.translation_log["frontmatter"]["target_language"] == "Spanish"


def test_translate_frontmatter_no_fields(translator_instance):
    """Test translate_frontmatter with no fields to translate."""
    # Call the method with empty fields list
    frontmatter_data = {
        "title": "Original Title",
        "description": "Original Description",
    }
    fields = []

    result, usage, error_msg = translator_instance.translate_frontmatter(
        frontmatter_data, fields, "Spanish", "gpt-4"
    )

    # Should return original data and empty usage stats
    assert result == frontmatter_data
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert error_msg is None