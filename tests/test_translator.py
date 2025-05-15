#!/usr/bin/env python3
# ABOUTME: Tests for the translator module.
# ABOUTME: Verifies translation functionality and API interactions.

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
    return Translator(openai_client)


def create_mock_response(content, usage=None):
    """Helper to create mock OpenAI API responses."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_usage = MagicMock()

    mock_message.content = content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    if usage is None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    mock_usage.prompt_tokens = usage["prompt_tokens"]
    mock_usage.completion_tokens = usage["completion_tokens"]
    mock_usage.total_tokens = usage["total_tokens"]
    mock_response.usage = mock_usage

    return mock_response


def test_init(openai_client):
    """Test translator initialization."""
    translator = Translator(openai_client)

    assert translator.client == openai_client
    assert isinstance(translator.translation_log, dict)
    assert "translation" in translator.translation_log
    assert "editing" in translator.translation_log
    assert "critique" in translator.translation_log
    assert "feedback" in translator.translation_log
    assert "frontmatter" in translator.translation_log
    assert "all_critiques" in translator.translation_log


def test_translate_text(translator_instance):
    """Test the translate_text method."""
    # Mock the OpenAI API response
    mock_response = create_mock_response("Texto traducido al español.")
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method
    text = "Text to be translated."
    target_language = "Spanish"
    model = "gpt-4"

    translated_text, usage = translator_instance.translate_text(
        text, target_language, model
    )

    # Verify results
    assert translated_text == "Texto traducido al español."
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30

    # Verify the client was called correctly
    translator_instance.client.chat.completions.create.assert_called_once()
    call_args = translator_instance.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1]["role"] == "user"

    # Verify translation log was updated
    assert "model" in translator_instance.translation_log["translation"]
    assert translator_instance.translation_log["translation"]["model"] == "gpt-4"
    assert "target_language" in translator_instance.translation_log["translation"]
    assert (
        translator_instance.translation_log["translation"]["target_language"]
        == "Spanish"
    )
    assert "response" in translator_instance.translation_log["translation"]
    assert "usage" in translator_instance.translation_log["translation"]


def test_translate_text_error(translator_instance):
    """Test error handling in translate_text method."""
    # Mock the OpenAI API to raise an exception
    translator_instance.client.chat.completions.create.side_effect = Exception(
        "API Error"
    )

    # Call the method and expect it to raise SystemExit
    with pytest.raises(SystemExit):
        translator_instance.translate_text("Text to translate", "Spanish", "gpt-4")


def test_edit_translation(translator_instance):
    """Test the edit_translation method."""
    # Mock the OpenAI API response
    mock_response = create_mock_response("Texto mejorado en español.")
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method
    translated_text = "Texto traducido al español."
    original_text = "Text to be translated."
    target_language = "Spanish"
    model = "gpt-4"

    edited_text, usage = translator_instance.edit_translation(
        translated_text, original_text, target_language, model
    )

    # Verify results
    assert edited_text == "Texto mejorado en español."
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30

    # Verify the client was called correctly
    translator_instance.client.chat.completions.create.assert_called_once()
    call_args = translator_instance.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1]["role"] == "user"
    assert "top_p" in call_args  # Should add top_p for non-o3 models

    # Verify editing log was updated
    assert "model" in translator_instance.translation_log["editing"]
    assert (
        translator_instance.translation_log["editing"]["target_language"] == "Spanish"
    )
    assert "response" in translator_instance.translation_log["editing"]
    assert "usage" in translator_instance.translation_log["editing"]


def test_edit_translation_with_o3_model(translator_instance):
    """Test edit_translation with the o3 model which requires special handling."""
    # Mock the OpenAI API response
    mock_response = create_mock_response("Texto mejorado con o3.")
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method with o3 model
    translated_text = "Texto traducido al español."
    original_text = "Text to be translated."
    target_language = "Spanish"
    model = "o3"

    edited_text, usage = translator_instance.edit_translation(
        translated_text, original_text, target_language, model
    )

    # Verify results
    assert edited_text == "Texto mejorado con o3."

    # Verify the client was called correctly - should NOT include top_p for o3
    call_args = translator_instance.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "o3"
    assert "top_p" not in call_args


def test_edit_translation_error(translator_instance):
    """Test error handling in edit_translation method."""
    # Mock the OpenAI API to raise an exception
    translator_instance.client.chat.completions.create.side_effect = Exception(
        "API Error"
    )

    # Call the method - should return original text and empty usage stats
    translated_text = "Texto original."
    result, usage = translator_instance.edit_translation(
        translated_text, "Original text", "Spanish", "gpt-4"
    )

    # Should return original text on error
    assert result == translated_text
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


def test_critique_translation(translator_instance):
    """Test the critique_translation method."""
    # Mock the OpenAI API response
    critique = "La traducción necesita mejoras en el tono y la naturalidad."
    mock_response = create_mock_response(critique)
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method
    translated_text = "Texto traducido al español."
    original_text = "Text to be translated."
    target_language = "Spanish"
    model = "gpt-4"

    result_text, usage, feedback = translator_instance.critique_translation(
        translated_text, original_text, target_language, model
    )

    # Verify results
    assert result_text == translated_text  # Critique doesn't change the text
    assert feedback == critique
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30

    # Verify the client was called correctly
    translator_instance.client.chat.completions.create.assert_called_once()
    call_args = translator_instance.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1]["role"] == "user"
    assert "temperature" in call_args  # Should add temperature for non-o3 models

    # Verify critique log was updated
    assert "model" in translator_instance.translation_log["critique"]
    assert (
        translator_instance.translation_log["critique"]["target_language"] == "Spanish"
    )
    assert "response" in translator_instance.translation_log["critique"]
    assert "usage" in translator_instance.translation_log["critique"]


def test_critique_translation_with_o3_model(translator_instance):
    """Test critique_translation with the o3 model which requires special handling."""
    # Mock the OpenAI API response
    critique = "Crítica generada con modelo o3."
    mock_response = create_mock_response(critique)
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method with o3 model
    translated_text = "Texto traducido."
    original_text = "Original text."
    target_language = "Spanish"
    model = "o3"

    _, _, feedback = translator_instance.critique_translation(
        translated_text, original_text, target_language, model
    )

    # Verify results
    assert feedback == critique

    # Verify the client was called correctly - should NOT include temperature for o3
    call_args = translator_instance.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "o3"
    assert "temperature" not in call_args


def test_critique_translation_error(translator_instance):
    """Test error handling in critique_translation method."""
    # Mock the OpenAI API to raise an exception
    translator_instance.client.chat.completions.create.side_effect = Exception(
        "API Error"
    )

    # Call the method - should return original text, empty usage stats, and empty critique
    translated_text = "Texto original."
    result, usage, critique = translator_instance.critique_translation(
        translated_text, "Original text", "Spanish", "gpt-4"
    )

    # Should return original text and empty critique on error
    assert result == translated_text
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert critique == ""


def test_apply_critique_feedback(translator_instance):
    """Test the apply_critique_feedback method."""
    # Mock the OpenAI API response
    improved_text = "Texto mejorado basado en la crítica."
    mock_response = create_mock_response(improved_text)
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method
    translated_text = "Texto traducido al español."
    original_text = "Text to be translated."
    critique = "La traducción necesita mejoras en fluidez."
    target_language = "Spanish"
    model = "gpt-4"

    result, usage = translator_instance.apply_critique_feedback(
        translated_text, original_text, critique, target_language, model
    )

    # Verify results
    assert result == improved_text
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30

    # Verify the client was called correctly
    translator_instance.client.chat.completions.create.assert_called_once()
    call_args = translator_instance.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert len(call_args["messages"]) == 2
    assert "temperature" in call_args  # Should add temperature for non-o3 models

    # Verify feedback log was updated
    assert "model" in translator_instance.translation_log["feedback"]
    assert (
        translator_instance.translation_log["feedback"]["target_language"] == "Spanish"
    )
    assert "response" in translator_instance.translation_log["feedback"]
    assert "usage" in translator_instance.translation_log["feedback"]


def test_apply_critique_feedback_with_o3_model(translator_instance):
    """Test apply_critique_feedback with the o3 model which requires special handling."""
    # Mock the OpenAI API response
    improved_text = "Texto mejorado con o3."
    mock_response = create_mock_response(improved_text)
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method with o3 model
    translated_text = "Texto traducido."
    original_text = "Original text."
    critique = "Needs improvement."
    target_language = "Spanish"
    model = "o3"

    result, _ = translator_instance.apply_critique_feedback(
        translated_text, original_text, critique, target_language, model
    )

    # Verify results
    assert result == improved_text

    # Verify the client was called correctly - should NOT include temperature for o3
    call_args = translator_instance.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "o3"
    assert "temperature" not in call_args


def test_apply_critique_feedback_error(translator_instance):
    """Test error handling in apply_critique_feedback method."""
    # Mock the OpenAI API to raise an exception
    translator_instance.client.chat.completions.create.side_effect = Exception(
        "API Error"
    )

    # Call the method - should return original text and empty usage stats
    translated_text = "Texto original."
    result, usage = translator_instance.apply_critique_feedback(
        translated_text, "Original text", "Critique", "Spanish", "gpt-4"
    )

    # Should return original text on error
    assert result == translated_text
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


def test_translate_frontmatter(translator_instance):
    """Test the translate_frontmatter method."""
    # Mock the OpenAI API response
    # Note: For this to work correctly with the regex extraction in the method,
    # we need to make sure each field appears on its own line or with proper formatting
    response_text = """
title: Título Traducido
description: Descripción traducida
    """
    mock_response = create_mock_response(response_text)
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method
    frontmatter_data = {
        "title": "Original Title",
        "description": "Original Description",
        "date": "2023-01-01",
        "author": "Author Name",
    }
    fields = ["title", "description"]
    target_language = "Spanish"
    model = "gpt-4"

    result, usage = translator_instance.translate_frontmatter(
        frontmatter_data, fields, target_language, model
    )

    # Verify results - should include translated and untranslated fields
    assert "Título Traducido" in result["title"]
    assert "Descripción traducida" in result["description"]
    assert result["date"] == "2023-01-01"  # Unchanged
    assert result["author"] == "Author Name"  # Unchanged

    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30

    # Verify the client was called correctly
    translator_instance.client.chat.completions.create.assert_called_once()

    # Verify frontmatter log was updated
    assert "model" in translator_instance.translation_log["frontmatter"]
    assert (
        translator_instance.translation_log["frontmatter"]["target_language"]
        == "Spanish"
    )
    assert "fields" in translator_instance.translation_log["frontmatter"]
    assert translator_instance.translation_log["frontmatter"]["fields"] == fields


def test_translate_frontmatter_no_fields(translator_instance):
    """Test translate_frontmatter with no fields to translate."""
    # Call the method with empty fields list
    frontmatter_data = {
        "title": "Original Title",
        "description": "Original Description",
    }
    fields = []

    result, usage = translator_instance.translate_frontmatter(
        frontmatter_data, fields, "Spanish", "gpt-4"
    )

    # Should return original data and empty usage stats
    assert result == frontmatter_data
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0

    # Should not call the API
    translator_instance.client.chat.completions.create.assert_not_called()


@patch("re.search")
def test_translate_frontmatter_pattern_matching(mock_re_search, translator_instance):
    """Test translate_frontmatter pattern matching for field extraction."""

    # Set up the mocks for regex pattern matching
    def mock_search_side_effect(pattern, text, flags):
        if "title:" in pattern:
            mock_match = MagicMock()
            mock_match.group.return_value = "Título con múltiples\nlíneas y formato"
            return mock_match
        elif "description:" in pattern:
            mock_match = MagicMock()
            mock_match.group.return_value = "Descripción\ncon saltos de línea"
            return mock_match
        return None

    mock_re_search.side_effect = mock_search_side_effect

    # Mock the API response
    response_text = """
    title: Título con múltiples
    líneas y formato

    description: Descripción
    con saltos de línea
    """
    mock_response = create_mock_response(response_text)
    translator_instance.client.chat.completions.create.return_value = mock_response

    # Call the method
    frontmatter_data = {
        "title": "Original Title",
        "description": "Original Description",
    }
    fields = ["title", "description"]

    result, _ = translator_instance.translate_frontmatter(
        frontmatter_data, fields, "Spanish", "gpt-4"
    )

    # Verify pattern matching extracted content correctly
    assert result["title"] == "Título con múltiples\nlíneas y formato"
    assert result["description"] == "Descripción\ncon saltos de línea"


def test_translate_frontmatter_error(translator_instance):
    """Test error handling in translate_frontmatter method."""
    # Mock the OpenAI API to raise an exception
    translator_instance.client.chat.completions.create.side_effect = Exception(
        "API Error"
    )

    # Call the method
    frontmatter_data = {"title": "Original Title"}
    fields = ["title"]

    result, usage = translator_instance.translate_frontmatter(
        frontmatter_data, fields, "Spanish", "gpt-4"
    )

    # Should return original data and empty usage stats
    assert result == frontmatter_data
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
