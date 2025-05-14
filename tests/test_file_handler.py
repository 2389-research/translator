#!/usr/bin/env python3
# ABOUTME: Tests for the file handler module.
# ABOUTME: Verifies file operations functionality.

import os
import pytest
import tempfile
from pathlib import Path
from translator.file_handler import FileHandler
from translator.language import LanguageHandler


@pytest.fixture
def temp_test_file():
    """Create a temporary test file for the tests."""
    temp_dir = tempfile.TemporaryDirectory()
    test_file_path = os.path.join(temp_dir.name, "test_file.txt")
    test_content = "This is test content for file operations."
    
    # Create a test file
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # Return the fixture data
    yield {
        "temp_dir": temp_dir,
        "test_file_path": test_file_path,
        "test_content": test_content
    }
    
    # Cleanup after tests
    temp_dir.cleanup()


def test_read_file(temp_test_file):
    """Test reading file content."""
    content = FileHandler.read_file(temp_test_file["test_file_path"])
    assert content == temp_test_file["test_content"]


def test_write_file(temp_test_file):
    """Test writing content to a file."""
    new_content = "This is new test content."
    new_file_path = os.path.join(temp_test_file["temp_dir"].name, "new_test_file.txt")
    
    FileHandler.write_file(new_file_path, new_content)
    
    # Verify the file was created and content is correct
    with open(new_file_path, 'r', encoding='utf-8') as f:
        assert f.read() == new_content


def test_get_output_filename_with_custom_output():
    """Test generating output filenames with custom output."""
    input_file = "/path/to/document.txt"
    target_language = "Spanish"
    custom_output = "/path/to/output/translated.txt"
    
    output_path = FileHandler.get_output_filename(input_file, target_language, custom_output)
    assert output_path == custom_output