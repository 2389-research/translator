#!/usr/bin/env python3
# ABOUTME: Test cases for the log interpreter module.
# ABOUTME: Verifies log file parsing and narrative generation.

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from translator.log_interpreter import LogInterpreter


class TestLogInterpreter(unittest.TestCase):
    """Test cases for the log interpreter."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.log_interpreter = LogInterpreter(self.mock_client)
        
        # Sample log data for testing
        self.sample_log_data = {
            "input_file": "test.md",
            "output_file": "test.fr.md",
            "target_language": "French",
            "language_code": "fr",
            "model": "o4",
            "skip_edit": False,
            "do_critique": True,
            "critique_loops": 2,
            "has_frontmatter": True,
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500
            },
            "cost": "$0.02",
            "prompts_and_responses": {
                "translation": {
                    "system_prompt": "You are a professional translator",
                    "user_prompt": "Translate this to French",
                    "response": "Content translated to French"
                },
                "editing": {
                    "system_prompt": "You are an editor",
                    "user_prompt": "Edit this translation",
                    "response": "Edited translation in French"
                },
                "critique": {
                    "system_prompt": "You are a critic",
                    "user_prompt": "Critique this translation",
                    "response": "Critique of the translation"
                },
                "all_critiques": [
                    "First critique",
                    "Second critique"
                ]
            }
        }

    def test_read_log_file(self):
        """Test reading and parsing a log file."""
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            json.dump(self.sample_log_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Test reading the file
            log_data = self.log_interpreter.read_log_file(temp_file_path)
            
            # Verify the content was read correctly
            self.assertEqual(log_data["target_language"], "French")
            self.assertEqual(log_data["language_code"], "fr")
            self.assertEqual(log_data["model"], "o4")
            self.assertEqual(log_data["token_usage"]["total_tokens"], 1500)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    def test_read_nonexistent_log_file(self):
        """Test reading a non-existent log file."""
        result = self.log_interpreter.read_log_file("/nonexistent/file.log")
        self.assertIsNone(result)
    
    def test_generate_narrative(self):
        """Test generating a narrative from log data."""
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a narrative interpretation of the translation process."
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Generate the narrative
        narrative = self.log_interpreter.generate_narrative(self.sample_log_data)
        
        # Verify the OpenAI API was called correctly
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "o4-mini")
        self.assertEqual(len(call_args["messages"]), 2)
        
        # Verify the narrative was generated
        self.assertEqual(narrative, "This is a narrative interpretation of the translation process.")
    
    def test_get_narrative_filename(self):
        """Test generating a narrative filename from a log file path."""
        log_path = "/path/to/file.fr.md.log"
        narrative_path = self.log_interpreter.get_narrative_filename(log_path)
        self.assertEqual(narrative_path, "/path/to/file.fr.md.narrative.md")
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_write_narrative(self, mock_open):
        """Test writing a narrative to a file."""
        narrative = "This is a narrative interpretation."
        output_path = "/path/to/output.narrative.md"
        
        self.log_interpreter.write_narrative(output_path, narrative)
        
        # Verify the file was opened and written to correctly
        mock_open.assert_called_once_with(output_path, 'w', encoding='utf-8')
        mock_open().write.assert_called_once_with(narrative)


if __name__ == '__main__':
    unittest.main()