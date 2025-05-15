# AICoder Translator 🚀

## Summary of Project

AICoder Translator is a powerful CLI tool that leverages OpenAI's API to translate text files into different languages while maintaining formatting and structure. It employs a sophisticated three-step process: initial translation, expert editing, and aggressive critique with revision cycles to produce high-quality translations.

The tool excels in handling various text formats, with special support for markdown files containing YAML frontmatter (commonly used in static site generators like Jekyll and Hugo). It preserves original formatting, structure, and markdown elements while providing natural-sounding translations in the target language.

Key features include:

- 🌐 Support for translating to any language supported by OpenAI models
- 📝 Preservation of original text formatting and structure
- 🔄 Multiple revision cycles for quality improvement
- 📊 Detailed logging and cost estimation
- 🧠 Intelligent handling of markdown frontmatter
- 📈 Token usage tracking and cost analysis

## How to Use

### Installation

1. Clone the repository

```bash
git clone https://github.com/2389-research/translator.git
cd translator
```

2. Configure your API key

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file to add your OpenAI API key
```

### Basic Usage

Translate a file to another language:

```bash
uv run main.py input.txt Spanish
```

This will create a translated file named `input.es.txt` in the same directory.

### Command Options

```bash
# Translate to French with a custom output file
uv run main.py README.md French -o translated_readme.md

# Use a specific OpenAI model
uv run main.py document.txt Japanese -m gpt-4o

# Skip the editing step for faster processing
uv run main.py long_document.txt German --no-edit

# Skip the critique step
uv run main.py quick_translation.txt Chinese --no-critique

# Specify number of critique-revision loops (1-5)
uv run main.py important_document.txt Korean --critique-loops 3

# View available models and pricing
uv run main.py --list-models

# Estimate cost without translating
uv run main.py large_document.txt Portuguese --estimate-only
```

## Tech Info

### Core Components

1. **Translation Pipeline**

    - **Frontmatter Handling**: Detects and processes YAML frontmatter in markdown files
    - **Content Translation**: Preserves formatting while translating the main content
    - **Editing Pass**: Ensures natural language and accurate translation
    - **Critique System**: Multiple rounds of critique and revision for higher quality

2. **Token Management**

    - Uses OpenAI's `tiktoken` library to count tokens
    - Checks document size against model limits
    - Estimates costs before translation

3. **Language Support**

    - Automatically generates ISO 639-1 language codes
    - Supports all languages available in OpenAI models
    - Uses `pycountry` library to map language names to codes

4. **Logging System**
    - Generates detailed JSON logs of the translation process
    - Records all prompts, responses, and metadata
    - Creates narrative summaries of the translation process

### Technical Architecture

The codebase is organized into modular components:

- `translator/`

    - `__init__.py`: Package initialization
    - `cli.py`: Command-line interface implementation
    - `config.py`: Model configuration and pricing information
    - `cost.py`: Cost estimation and calculation
    - `file_handler.py`: File I/O utilities
    - `frontmatter_handler.py`: Processes YAML frontmatter
    - `language.py`: Language code detection and mapping
    - `log_interpreter.py`: Analyzes and creates narratives from logs
    - `prompts.py`: Centralized storage for system and user prompts
    - `token_counter.py`: Token counting functions
    - `translator.py`: Core translation logic

- `tests/`: Comprehensive test suite
- `samples/`: Example files for testing

### Development Notes

The project uses pytest for testing. To run tests:

```bash
uv run pytest
```

The prompting system is designed for high-quality translations with multiple improvement stages:

1. Initial translation with formatting preservation
2. Expert editing for natural language
3. Critical analysis and feedback
4. Application of critique for final polishing

This multi-stage approach produces translations that read as if they were originally written in the target language while maintaining fidelity to the source.

### Technical Requirements

- Python 3.13+
- Dependencies (automatically installed):
    - openai>=1.78.1
    - python-dotenv>=1.1.0
    - rich>=13.9.4
    - tiktoken>=0.7.0
    - pycountry>=23.12.10
    - python-frontmatter>=1.1.0
    - pytest>=7.4.0 (for testing)

The tool is designed to be extended with new models and features as OpenAI's API evolves.
