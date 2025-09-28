# Translator CLI Tool - Technical Specification

## Project Overview

The Translator CLI Tool is a Python-based command-line application that leverages OpenAI's API to translate text files into different languages while maintaining formatting and structure. The tool employs a sophisticated multi-stage translation process with expert editing and critique cycles to produce high-quality translations.

**Version:** 0.3.0
**Python Requirement:** >= 3.13
**License:** Open Source
**Repository:** [2389-research/translator](https://github.com/2389-research/translator)

## Architecture Overview

The system follows a clean layered architecture with clear separation of concerns:

```
User Input → Entry Point → CLI Orchestrator → Core Engine + Utilities → Output
```

### Architectural Layers

1. **Entry Point Layer** - Application bootstrap and initialization
2. **CLI Interface Layer** - User interaction and workflow orchestration
3. **Core Translation Engine** - Translation logic and prompt management
4. **Configuration & Cost Management** - Model settings and pricing
5. **File & Content Processing** - I/O operations and format handling
6. **Logging & Analysis** - Process tracking and reporting
7. **External Dependencies** - Third-party integrations

## Core Components

### 1. Entry Point Layer

#### `main.py`
- **Purpose:** CLI Entry Point
- **Functionality:** Simple bootstrap that delegates to TranslatorCLI
- **Dependencies:** translator.cli.TranslatorCLI
- **Interface:** Command-line arguments

### 2. CLI Interface Layer

#### `cli.py` - TranslatorCLI (Orchestrator)
- **Purpose:** Central workflow orchestrator and user interface
- **Key Responsibilities:**
  - Command-line argument parsing
  - Workflow coordination across all modules
  - User interface rendering (Rich terminal UI)
  - Error handling and user feedback
  - Configuration management
- **Dependencies:** ALL internal modules + external UI libraries
- **Architecture Role:** Fan-out coordinator with high coupling (appropriate for CLI)

### 3. Core Translation Engine

#### `translator.py` - Translator (Core Logic)
- **Purpose:** Core translation business logic
- **Key Responsibilities:**
  - OpenAI API integration
  - Multi-stage translation process (translate → edit → critique → revise)
  - Streaming response handling
  - Error recovery and retry logic
- **Dependencies:** prompts.py, openai, swarm
- **Interface:** Translation methods with callback support

#### `prompts.py` - Prompts (Templates)
- **Purpose:** Centralized prompt template management
- **Key Responsibilities:**
  - System and user prompt definitions
  - Template parameterization
  - Prompt versioning and variation
- **Dependencies:** None (pure templates)
- **Architecture Role:** Template provider for core logic

### 4. Configuration & Cost Management

#### `config.py` - ModelConfig (Models & Settings)
- **Purpose:** Model configuration and settings management
- **Key Responsibilities:**
  - OpenAI model definitions and capabilities
  - Model-specific parameter management
  - Configuration validation
- **Dependencies:** None
- **Interface:** Configuration data classes and utilities

#### `cost.py` - CostEstimator (Price Calculation)
- **Purpose:** Translation cost estimation and tracking
- **Key Responsibilities:**
  - Token-based cost calculation
  - Model pricing database
  - Cost reporting and budgeting
- **Dependencies:** config.py (for model pricing)
- **Interface:** Cost estimation methods

#### `token_counter.py` - TokenCounter (Token Counting)
- **Purpose:** Accurate token counting for OpenAI models
- **Key Responsibilities:**
  - Model-specific tokenization
  - Token limit validation
  - Streaming token tracking
- **Dependencies:** config.py (for model info), tiktoken
- **Interface:** Token counting utilities

### 5. File & Content Processing

#### `file_handler.py` - FileHandler (I/O Operations)
- **Purpose:** File system operations and content management
- **Key Responsibilities:**
  - File reading and writing
  - Output filename generation
  - Directory management
  - Error handling for file operations
- **Dependencies:** language.py (for language codes), rich (for UI)
- **Interface:** File operation methods

#### `frontmatter_handler.py` - FrontmatterHandler (YAML Processing)
- **Purpose:** Markdown frontmatter processing
- **Key Responsibilities:**
  - YAML frontmatter extraction and preservation
  - Metadata handling for static site generators
  - Content and metadata separation
- **Dependencies:** python-frontmatter, rich (for UI)
- **Interface:** Frontmatter processing methods

#### `language.py` - LanguageHandler (Language Codes)
- **Purpose:** Language detection and code mapping
- **Key Responsibilities:**
  - Language name to ISO 639-1 code conversion
  - Supported language validation
  - Internationalization support
- **Dependencies:** pycountry
- **Interface:** Language utility methods

### 6. Logging & Analysis

#### `log_interpreter.py` - LogInterpreter (Process Analysis)
- **Purpose:** Translation process analysis and reporting
- **Key Responsibilities:**
  - JSON log generation and parsing
  - Process narrative creation
  - Performance analysis
  - Quality metrics tracking
- **Dependencies:** openai (for analysis), rich (for UI)
- **Interface:** Logging and analysis methods

### 7. External Dependencies

#### Core API Integration
- **openai** (>=1.78.1) - Primary AI API client
- **swarm** (from GitHub) - OpenAI extensions (future use)

#### User Interface
- **rich** (>=13.9.4) - Terminal UI framework
  - Console output formatting
  - Progress indicators
  - Table rendering
  - Error display

#### Content Processing
- **python-frontmatter** (>=1.1.0) - YAML frontmatter parsing
- **tiktoken** (>=0.7.0) - OpenAI tokenization
- **pycountry** (>=23.12.10) - Country and language codes

#### Configuration
- **python-dotenv** (>=1.1.0) - Environment variable loading

#### Development & Testing
- **pytest** (>=7.4.0) - Test framework
- **pytest-cov** (>=6.1.1) - Coverage reporting

## Data Flow Architecture

### Primary Workflow
```
User Input (file + target language)
    ↓
main.py (bootstrap)
    ↓
cli.py (argument parsing & orchestration)
    ↓ (parallel operations)
    ├── file_handler.py (read source file)
    ├── frontmatter_handler.py (extract metadata)
    ├── token_counter.py (count tokens)
    ├── cost.py (estimate cost)
    └── config.py (load model settings)
    ↓
translator.py (core translation process)
    ├── prompts.py (load templates)
    ├── openai API (translation calls)
    └── log_interpreter.py (process logging)
    ↓
cli.py (result processing)
    ├── file_handler.py (write output)
    └── rich UI (display results)
    ↓
Output (translated file + cost report + logs)
```

### Translation Process Stages

1. **Preparation Phase**
   - File reading and validation
   - Frontmatter extraction (for markdown)
   - Token counting and cost estimation
   - Model configuration loading

2. **Translation Phase**
   - Initial translation with formatting preservation
   - Streaming response handling
   - Progress tracking and user feedback

3. **Enhancement Phase** (optional)
   - Expert editing for natural language flow
   - Critical analysis and feedback generation
   - Iterative revision cycles (1-5 loops)

4. **Output Phase**
   - File writing with appropriate naming
   - Cost reporting and analysis
   - Process logging and narrative generation

## Interface Specifications

### Command Line Interface

#### Primary Commands
```bash
translator <input_file> <target_language> [options]
translator translate <input_file> <target_language> [options]
translator config                    # Interactive configuration
translator --list-models            # Display available models
translator --estimate-only          # Cost estimation only
```

#### Options
- `-o, --output <file>` - Custom output filename
- `-m, --model <model>` - Specify OpenAI model
- `--no-edit` - Skip editing phase
- `--no-critique` - Skip critique phase
- `--critique-loops <n>` - Number of critique iterations (1-5)

#### Environment Variables
- `OPENAI_API_KEY` - Required API key
- `DEFAULT_MODEL` - Default model selection
- `OUTPUT_DIR` - Default output directory
- `LOG_LEVEL` - Logging verbosity

### Programming Interface

#### Core Translation API
```python
from translator.translator import Translator
from translator.config import ModelConfig

translator = Translator()
result = translator.translate(
    text="Source text",
    target_language="Spanish",
    model="gpt-4o",
    streaming=True,
    token_callback=callback_fn
)
```

#### File Processing API
```python
from translator.file_handler import FileHandler
from translator.frontmatter_handler import FrontmatterHandler

file_handler = FileHandler()
content = file_handler.read_file("input.md")

frontmatter_handler = FrontmatterHandler()
metadata, content = frontmatter_handler.extract_frontmatter(content)
```

## Configuration Management

### Configuration Sources (Priority Order)
1. Environment variables
2. Local `.env` file
3. User configuration directory (`~/.translator/.env`)
4. Alternative user directory (`~/.config/translator/.env`)

### Supported Models
- **o3** - Advanced reasoning model (default)
- **gpt-4o** - Optimized GPT-4 variant
- **gpt-4** - Standard GPT-4 model
- Additional models as supported by OpenAI API

### Cost Management
- Real-time token counting
- Pre-translation cost estimation
- Model-specific pricing database
- Usage tracking and reporting

## Quality Assurance

### Testing Strategy
- **Unit Tests** - Individual module testing
- **Integration Tests** - Cross-module functionality
- **End-to-End Tests** - Complete workflow validation
- **Coverage Target** - >90% code coverage

### Test Infrastructure
```
tests/
├── test_translator.py      # Core logic tests
├── test_cli.py            # CLI interface tests
├── test_file_handler.py   # File operations tests
├── test_frontmatter_handler.py  # YAML processing tests
├── test_cost.py           # Cost calculation tests
├── test_token_counter.py  # Token counting tests
├── test_config.py         # Configuration tests
├── test_language.py       # Language handling tests
├── test_log_interpreter.py # Logging tests
└── test_streaming.py      # Streaming functionality tests
```

### Code Quality Standards
- **Type Annotations** - Comprehensive type hints
- **Documentation** - Docstring coverage for public APIs
- **Linting** - Code style enforcement
- **Pre-commit Hooks** - Automated quality checks

## Performance Characteristics

### Scalability
- **File Size Limits** - Constrained by OpenAI token limits
- **Concurrent Operations** - Single-threaded with async potential
- **Memory Usage** - Efficient streaming for large files
- **API Rate Limits** - Built-in retry and backoff logic

### Optimization Features
- **Streaming Responses** - Real-time translation display
- **Token Efficiency** - Accurate counting and optimization
- **Caching** - None currently (future enhancement)
- **Batch Processing** - Single file focus (future enhancement)

## Security Considerations

### API Key Management
- Environment variable preferred
- File-based configuration with appropriate permissions
- No hard-coded credentials
- Clear documentation of security practices

### Input Validation
- File path validation and sanitization
- Content size limits
- Language code validation
- Model parameter validation

### Output Security
- Safe file writing practices
- Directory traversal prevention
- Temporary file cleanup
- Error message sanitization

## Deployment & Distribution

### Package Management
- **Build System** - UV-based Python packaging
- **Dependencies** - Lockfile-managed with uv.lock
- **Distribution** - PyPI-compatible wheel packages
- **Installation** - `uv tool install` support

### Platform Support
- **Operating Systems** - Cross-platform (Windows, macOS, Linux)
- **Python Versions** - 3.13+ required
- **Architecture** - x86_64, ARM64 compatible

### Configuration Files
```
pyproject.toml          # Package metadata and dependencies
uv.lock                 # Dependency lockfile
pytest.ini              # Test configuration
.pre-commit-config.yaml # Code quality automation
.env.example            # Configuration template
```

## Extension Points

### Future Enhancements
1. **Multi-file Processing** - Batch translation support
2. **Additional AI Providers** - Anthropic, local models
3. **Translation Memory** - Caching and consistency
4. **API Mode** - Web service deployment
5. **Plugin System** - Custom prompt providers
6. **Advanced Formatting** - PDF, HTML support

### Architecture Support
- **Prompt Providers** - Pluggable template system
- **Output Formatters** - Multiple format support
- **Model Adapters** - Provider abstraction layer
- **Configuration Providers** - Alternative config sources

## Monitoring & Observability

### Logging Infrastructure
- **Structured Logging** - JSON format with metadata
- **Process Tracking** - Complete translation workflow
- **Performance Metrics** - Token usage, timing, costs
- **Error Reporting** - Detailed error context

### Analytics Capabilities
- **Cost Analysis** - Per-translation and aggregate costs
- **Quality Metrics** - Translation accuracy tracking
- **Usage Patterns** - Model and feature utilization
- **Performance Profiling** - Bottleneck identification

## Technical Debt & Known Issues

### Current Limitations (from issues.md analysis)
1. **Type Checking** - Missing type annotations in callback functions
2. **Token Counting** - Inefficient streaming mode counting
3. **Memory Management** - Potential leaks in cancellation handler
4. **Code Duplication** - Redundant model validation logic
5. **Hard-coded Parameters** - API parameters should be configurable
6. **Error Handling** - Inconsistent patterns across modules
7. **Method Complexity** - Excessive nesting in CLI module
8. **Documentation** - Missing docstrings for public APIs
9. **Concurrency** - Potential race conditions in streaming
10. **Configuration** - Hard-coded paths and defaults

### Improvement Roadmap
- Comprehensive type annotation addition
- Streaming token counting optimization
- Error handling standardization
- Method decomposition and refactoring
- Configuration externalization
- Documentation completion
- Concurrency safety improvements

---

*This specification is generated from the architectural analysis of translator_architecture.dot and represents the current state of the system as of version 0.3.0.*