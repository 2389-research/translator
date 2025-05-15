#!/usr/bin/env python3
# ABOUTME: Tests for the file handler module.
# ABOUTME: Verifies file operations functionality.

import os
import pytest
import tempfile
from unittest.mock import patch
from translator.file_handler import FileHandler
from translator.language import LanguageHandler


@pytest.fixture
def temp_test_file():
    """Create a temporary test file for the tests."""
    temp_dir = tempfile.TemporaryDirectory()
    test_file_path = os.path.join(temp_dir.name, "test_file.txt")
    test_content = "This is test content for file operations."

    # Create a test file
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)

    # Return the fixture data
    yield {
        "temp_dir": temp_dir,
        "test_file_path": test_file_path,
        "test_content": test_content,
    }

    # Cleanup after tests
    temp_dir.cleanup()


def test_read_file(temp_test_file):
    """Test reading file content."""
    content = FileHandler.read_file(temp_test_file["test_file_path"])
    assert content == temp_test_file["test_content"]


def test_read_file_error():
    """Test error handling when reading a non-existent file."""
    nonexistent_file = "/path/to/nonexistent/file.txt"
    with pytest.raises(SystemExit):
        FileHandler.read_file(nonexistent_file)


def test_write_file(temp_test_file):
    """Test writing content to a file."""
    new_content = "This is new test content."
    new_file_path = os.path.join(temp_test_file["temp_dir"].name, "new_test_file.txt")

    FileHandler.write_file(new_file_path, new_content)

    # Verify the file was created and content is correct
    with open(new_file_path, "r", encoding="utf-8") as f:
        assert f.read() == new_content


def test_write_file_error():
    """Test error handling when writing to an invalid location."""
    with pytest.raises(SystemExit):
        FileHandler.write_file("/invalid/directory/file.txt", "Test content")


def test_get_output_filename_with_custom_output():
    """Test generating output filenames with custom output."""
    input_file = "/path/to/document.txt"
    target_language = "Spanish"
    custom_output = "/path/to/output/translated.txt"

    output_path = FileHandler.get_output_filename(
        input_file, target_language, custom_output
    )
    assert output_path == custom_output


def test_get_output_filename_without_custom_output():
    """Test generating output filenames without custom output."""
    input_file = "/path/to/document.txt"
    target_language = "Spanish"

    with patch.object(LanguageHandler, "get_language_code", return_value="es"):
        output_path = FileHandler.get_output_filename(input_file, target_language)
        assert output_path == "/path/to/document.es.txt"


def test_get_output_filename_different_languages():
    """Test generating output filenames for different languages."""
    input_file = "/path/to/document.txt"

    with patch.object(LanguageHandler, "get_language_code") as mock_get_language_code:
        # Test with Japanese
        mock_get_language_code.return_value = "ja"
        output_path = FileHandler.get_output_filename(input_file, "Japanese")
        assert output_path == "/path/to/document.ja.txt"

        # Test with French
        mock_get_language_code.return_value = "fr"
        output_path = FileHandler.get_output_filename(input_file, "French")
        assert output_path == "/path/to/document.fr.txt"


def test_write_log():
    """Test writing log data to a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(temp_dir, "test.log")
        log_data = {"translation": "Test content", "model": "gpt-4"}

        FileHandler.write_log(log_path, log_data)

        # Verify the log file was created
        assert os.path.exists(log_path)

        # Read and verify content (basic check)
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Test content" in content
            assert "gpt-4" in content
            assert "timestamp" in content


def test_write_log_error():
    """Test error handling when writing log fails."""
    with patch("builtins.open", side_effect=Exception("Test error")):
        # Should not raise SystemExit, just print a warning
        FileHandler.write_log("/path/to/log.log", {"test": "data"})


def test_get_log_filename():
    """Test generating log filename based on output file."""
    output_file = "/path/to/document.es.txt"
    log_path = FileHandler.get_log_filename(output_file)
    assert log_path == "/path/to/document.es.txt.log.json"


def test_get_log_filename_different_extensions():
    """Test generating log filename with different file extensions."""
    # Test with markdown file
    output_file = "/path/to/document.ja.md"
    log_path = FileHandler.get_log_filename(output_file)
    assert log_path == "/path/to/document.ja.md.log.json"

    # Test with json file
    output_file = "/path/to/data.es.json"
    log_path = FileHandler.get_log_filename(output_file)
    assert log_path == "/path/to/data.es.json.log.json"
