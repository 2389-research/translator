# Translator CLI

A simple command-line tool to translate text files to different languages using OpenAI's API.

## Features

- Translate text files (`.txt`, `.md`, etc.) to any language
- Three-step process by default:
  - Initial translation preserving structure and formatting
  - Expert editing for natural-sounding language
  - Aggressive critique and revision for highest quality results
- Special handling for markdown blog posts with YAML frontmatter (Jekyll, Hugo, etc.)
- Preserves original formatting and markdown structure
- Automatically generates output filenames with ISO language codes
- Configurable OpenAI model selection
- Detailed translation logs with all prompts and responses

## Installation

1. Clone this repository
2. Install with UV:
   ```bash
   # Create a virtual environment (optional but recommended)
   uv venv

   # Install the package in development mode
   uv pip install -e .
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Usage

Basic usage:

```bash
translate input.txt Spanish
```

This will translate `input.txt` to Spanish and save it as `input.es.txt`.

### Options

- `-o, --output`: Specify custom output file
- `-m, --model`: Choose OpenAI model (default: o3)
- `--no-edit`: Skip the editing step (faster but may reduce quality)
- `--no-critique`: Skip the aggressive critique step (faster but may reduce quality)
- `--critique-loops`: Number of critique-revision loops to perform (default: 4, max: 5)
- `--list-models`: Display available models and their pricing
- `--estimate-only`: Estimate token usage and cost without translating

Examples:

```bash
# Translate to French with custom output file
translate README.md French -o translated_readme.md

# Translate using a different model
translate document.txt Japanese -m gpt-4o

# Skip the editing step for faster processing
translate long_document.txt German --no-edit

# Skip the critique step for faster processing
translate quick_translation.txt Chinese --no-critique

# Use 4 critique loops for highest quality translation
translate important_document.txt Korean --critique-loops 4

# See available models and pricing
translate --list-models

# Estimate the cost before translating
translate large_document.txt Portuguese --estimate-only
```

## How It Works

1. **Frontmatter Detection**: For markdown files, detects and parses frontmatter using python-frontmatter (supports YAML format)
2. **Frontmatter Translation**: Translates metadata fields like title, description, and summary
3. **Content Translation**: Translates the main content to the target language while preserving formatting
4. **Content Editing** (optional): A second pass is made with both the original and translated text to:
   - Compare with the original to ensure accurate translation
   - Fix grammatical errors
   - Make text sound natural to native speakers
   - Adjust idioms and expressions to be appropriate for the target language
   - Maintain the original meaning, tone, and nuance
5. **Aggressive Critique** (optional): A two-step process for achieving the highest quality:
   - **Critique Generation**: A highly critical translator analyzes the translation and provides detailed feedback on issues
     • Meticulously compares the translated text with the original
     • Identifies any inaccuracies, mistranslations, or omissions
     • Scrutinizes for awkward phrasing or unnatural expressions
     • Lists specific issues organized by severity and category
   - **Critique Application**: A master translator applies the critique feedback
     • Addresses all identified issues with precision
     • Implements specific suggestions for improvement
     • Refines the translation to read as if originally written in the target language
6. **Log Generation**: Creates a detailed log file with all translation steps:
   - Saved as a JSON file next to the translated file (filename.language.ext.log)
   - Contains all prompts, responses, and metadata from each step of the process
   - Useful for debugging, improving prompts, or learning from the translation

## Supported Languages

The translator can work with any language supported by OpenAI models. It intelligently generates ISO 639-1 two-letter language codes for output filenames:

1. First checks against a built-in list of common languages
2. Uses `pycountry` library to look up standard codes
3. Falls back to the first two letters of the language name if no match is found

Some examples:
- "Spanish" → `es`
- "French" → `fr`
- "Mandarin" → `zh`
- "Brazilian Portuguese" → `pt`
- "Modern Greek" → `el`

## Token Usage and Costs

The tool provides both estimated and actual token usage and cost information:

### Before Translation
- Estimates token count using OpenAI's `tiktoken` library
- Estimates cost based on current OpenAI pricing
- Warns if document may exceed model token limits
- Use `--estimate-only` to check estimates without translating

### After Translation
- Displays actual token usage from the OpenAI API
- Shows detailed breakdown by operation (translation, editing, critique)
- Calculates actual cost based on tokens used
- When using `--critique`, shows both critique generation and application costs
- Use `--list-models` to view all supported models with their limits and pricing

The pricing table includes the latest rates for all supported models (as of May 2025), from affordable options like `gpt-4o-mini` to high-end models like `o1-pro`.

## Code Structure

The translator is organized into modular components:

- `config.py`: Model configurations and pricing
- `token_counter.py`: Token counting utilities
- `cost.py`: Cost estimation and calculation
- `language.py`: Language code handling
- `file_handler.py`: File I/O operations
- `frontmatter_handler.py`: Frontmatter processing
- `translator.py`: Core translation logic
- `cli.py`: Command-line interface

This structure makes the code more maintainable and extendable.

## Development

### Running Tests

The project uses pytest for testing. To run tests:

```bash
# Run all tests
./run_tests.sh

# Run specific tests
./run_tests.sh tests/test_token_counter.py
```

See the [tests README](tests/README.md) for more information.

## License

MIT