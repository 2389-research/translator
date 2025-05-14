# Translator Tests

This directory contains tests for the Translator package using pytest.

## Running Tests

Using the provided test script:

```bash
# Run all tests
./run_tests.sh

# Run specific test files
./run_tests.sh tests/test_token_counter.py

# Run with specific pytest options
./run_tests.sh -v --durations=5
```

Or run pytest manually:

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_token_counter.py

# Run a specific test function
uv run pytest tests/test_token_counter.py::test_count_tokens

# Show test durations
uv run pytest --durations=5
```

## Test Structure

- `test_token_counter.py`: Tests for token counting functionality
- `test_language.py`: Tests for language code detection
- `test_file_handler.py`: Tests for file I/O operations

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new file named `test_<module>.py` for each module being tested
2. Use the `unittest` framework for consistency
3. Add ABOUTME header comments at the top of each test file
4. Include both positive and negative test cases
5. Mock external dependencies when appropriate