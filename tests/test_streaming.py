#!/usr/bin/env python3
# ABOUTME: Unit tests for streaming functionality.
# ABOUTME: Tests the streaming implementation of the OpenAI API.

import unittest
from unittest.mock import Mock, patch
from translator.translator import Translator
from translator.cli import CancellationHandler


class TestStreaming(unittest.TestCase):
    """Test cases for streaming functionality."""

    @patch("openai.OpenAI")
    def test_translate_text_streaming(self, mock_openai):
        """Test that translate_text works with streaming."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock the streaming response
        mock_response = []
        # Create mock chunks
        for i in range(3):
            mock_chunk = Mock()
            mock_choice = Mock()
            mock_delta = Mock()
            mock_delta.content = f"chunk{i} "
            mock_choice.delta = mock_delta
            mock_choice.index = 0
            mock_chunk.choices = [mock_choice]
            mock_response.append(mock_chunk)

        # Set up the mock chat completions create method
        mock_client.chat.completions.create.return_value = mock_response

        # Create a translator
        translator = Translator(mock_client)

        # Call translate_text with streaming
        translated_text, usage, error_msg = translator.translate_text(
            "Hello world", "French", "o3", stream=True
        )

        # Check the results
        self.assertEqual(translated_text, "chunk0 chunk1 chunk2 ")
        self.assertIsNotNone(usage)
        self.assertIsNone(error_msg)
        mock_client.chat.completions.create.assert_called_with(
            model="o3",
            messages=[
                {"role": "system", "content": unittest.mock.ANY},
                {"role": "user", "content": unittest.mock.ANY},
            ],
            stream=True
        )

    @patch("openai.OpenAI")
    def test_cancellation_while_streaming(self, mock_openai):
        """Test cancellation during streaming."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Create a custom mock response generator that can be cancelled
        class MockStreamResponse:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= len(self.chunks):
                    raise StopIteration
                result = self.chunks[self.index]
                self.index += 1
                return result

        # Create mock chunks
        chunks = []
        for i in range(5):
            mock_chunk = Mock()
            mock_choice = Mock()
            mock_delta = Mock()
            mock_delta.content = f"chunk{i} "
            mock_choice.delta = mock_delta
            mock_choice.index = 0
            mock_chunk.choices = [mock_choice]
            chunks.append(mock_chunk)

        # Set up the mock chat completions create method
        mock_client.chat.completions.create.return_value = MockStreamResponse(chunks)

        # Create a translator and a cancellation handler
        translator = Translator(mock_client)
        cancellation = CancellationHandler()

        # Set up cancellation after the second chunk
        original_is_cancellation_requested = cancellation.is_cancellation_requested
        call_count = [0]  # Use a mutable object to track calls

        def mock_is_cancellation_requested():
            call_count[0] += 1
            # Return True after the second call to simulate cancellation after 2 chunks
            return call_count[0] > 2

        cancellation.is_cancellation_requested = mock_is_cancellation_requested

        # Call translate_text with streaming and cancellation
        translated_text, usage, error_msg = translator.translate_text(
            "Hello world", "French", "o3", stream=True, cancellation_handler=cancellation
        )

        # Restore the original method
        cancellation.is_cancellation_requested = original_is_cancellation_requested

        # Check the results - should only have the first 2 chunks
        self.assertEqual(translated_text, "chunk0 chunk1 ")
        self.assertIsNotNone(usage)
        self.assertIsNone(error_msg)
        mock_client.chat.completions.create.assert_called_with(
            model="o3",
            messages=[
                {"role": "system", "content": unittest.mock.ANY},
                {"role": "user", "content": unittest.mock.ANY},
            ],
            stream=True
        )


if __name__ == "__main__":
    unittest.main()