# Translator CLI

A simple command-line tool to translate text files to different languages using OpenAI's API.

## Features

- Translate text files (`.txt`, `.md`, etc.) to any language
- Two-step process: translation followed by expert editing for natural results
- Special handling for markdown blog posts with YAML frontmatter (Jekyll, Hugo, etc.)
- Preserves original formatting and markdown structure
- Automatically generates output filenames with ISO language codes
- Configurable OpenAI model selection

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   uv add openai python-dotenv rich
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key

## Usage

Basic usage:

```bash
uv run main.py input.txt Spanish
```

This will translate `input.txt` to Spanish and save it as `input.es.txt`.

### Options

- `-o, --output`: Specify custom output file
- `-m, --model`: Choose OpenAI model (default: o3)
- `--no-edit`: Skip the editing step (faster but may reduce quality)
- `--list-models`: Display available models and their pricing
- `--estimate-only`: Estimate token usage and cost without translating

Examples:

```bash
# Translate to French with custom output file
uv run main.py README.md French -o translated_readme.md

# Translate using a different model
uv run main.py document.txt Japanese -m gpt-4o

# Skip the editing step for faster processing
uv run main.py long_document.txt German --no-edit

# See available models and pricing
uv run main.py --list-models

# Estimate the cost before translating
uv run main.py large_document.txt Portuguese --estimate-only
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
- Shows detailed breakdown by operation (translation and editing)
- Calculates actual cost based on tokens used
- Use `--list-models` to view all supported models with their limits and pricing

The pricing table includes the latest rates for all supported models (as of May 2025), from affordable options like `gpt-4o-mini` to high-end models like `o1-pro`.

## License

MIT