#!/usr/bin/env python3
# ABOUTME: Command-line interface for translating text files to different languages.
# ABOUTME: Uses OpenAI's API to perform translations while maintaining formatting.

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import frontmatter
import openai
import pycountry
import tiktoken
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Model configuration: max_tokens and cost per 1k tokens (input/output)
MODEL_CONFIG = {
    "o3": {
        "max_tokens": 128000,
        "input_cost": 0.01,   # $10.00 / 1M tokens
        "output_cost": 0.04   # $40.00 / 1M tokens
    },
    "gpt-4o": {
        "max_tokens": 128000,
        "input_cost": 0.005,  # $5.00 / 1M tokens
        "output_cost": 0.02   # $20.00 / 1M tokens
    },
    "gpt-4o-mini": {
        "max_tokens": 128000,
        "input_cost": 0.00015,  # $0.15 / 1M tokens
        "output_cost": 0.0006   # $0.60 / 1M tokens
    },
    "gpt-4-turbo": {
        "max_tokens": 128000,
        "input_cost": 0.01,   # $10.00 / 1M tokens
        "output_cost": 0.03   # $30.00 / 1M tokens
    },
    "gpt-4": {
        "max_tokens": 8192,
        "input_cost": 0.03,   # $30.00 / 1M tokens
        "output_cost": 0.06   # $60.00 / 1M tokens
    },
    "gpt-3.5-turbo": {
        "max_tokens": 16385,
        "input_cost": 0.0005,  # $0.50 / 1M tokens
        "output_cost": 0.0015  # $1.50 / 1M tokens
    },
    "gpt-4-vision-preview": {
        "max_tokens": 128000,
        "input_cost": 0.01,   # $10.00 / 1M tokens
        "output_cost": 0.03   # $30.00 / 1M tokens
    },
    "gpt-4.1": {
        "max_tokens": 1048576,
        "input_cost": 0.002,  # $2.00 / 1M tokens
        "output_cost": 0.008  # $8.00 / 1M tokens
    },
    "gpt-4.1-mini": {
        "max_tokens": 1048576,
        "input_cost": 0.0004,  # $0.40 / 1M tokens
        "output_cost": 0.0016  # $1.60 / 1M tokens
    },
    "gpt-4.1-nano": {
        "max_tokens": 1048576,
        "input_cost": 0.0001,  # $0.10 / 1M tokens
        "output_cost": 0.0004  # $0.40 / 1M tokens
    },
    "o1-pro": {
        "max_tokens": 200000,
        "input_cost": 0.15,   # $150.00 / 1M tokens
        "output_cost": 0.60   # $600.00 / 1M tokens
    }
}


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text string for a specific model."""
    try:
        # Use gpt-4 encoder for o3 model
        model_name = model
        if model == "o3":
            model_name = "gpt-4"
            
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to cl100k_base if model-specific encoding not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def check_token_limits(content: str, model: str) -> Tuple[bool, int]:
    """Check if content is within token limits for the model."""
    token_count = count_tokens(content, model)
    
    # Get max tokens for the model, default to 4000 if not found
    max_tokens = MODEL_CONFIG.get(model, {}).get("max_tokens", 4000)
    
    # We need room for system prompt, translation, and response
    # Estimate total tokens needed as 2.5x the input content
    estimated_total = token_count * 2.5
    
    return (estimated_total <= max_tokens, token_count)

def estimate_cost(token_count: int, model: str, with_edit: bool = True) -> Tuple[float, str]:
    """Estimate the cost of translation based on token count and model."""
    if model not in MODEL_CONFIG:
        return (0.0, "Unknown model, cost cannot be estimated")
    
    # Get cost per 1k tokens
    input_cost = MODEL_CONFIG[model]["input_cost"]
    output_cost = MODEL_CONFIG[model]["output_cost"]
    
    # Estimate input/output tokens
    input_tokens = token_count
    # Translation typically produces similar token count to original
    output_tokens = token_count
    
    # If editing is enabled, we'll make a second API call
    if with_edit:
        # For the edit, we input both original and translated text
        edit_input_tokens = token_count * 2
        # Output is similar to the translation
        edit_output_tokens = token_count
        
        # Calculate total cost in USD
        cost = (
            (input_tokens / 1000) * input_cost +
            (output_tokens / 1000) * output_cost +
            (edit_input_tokens / 1000) * input_cost +
            (edit_output_tokens / 1000) * output_cost
        )
    else:
        # Calculate translation-only cost
        cost = (
            (input_tokens / 1000) * input_cost +
            (output_tokens / 1000) * output_cost
        )
    
    # Format approximate price
    if cost < 0.01:
        cost_str = f"Less than $0.01"
    else:
        cost_str = f"Approximately ${cost:.2f}"
    
    return (cost, cost_str)

def setup_openai_client() -> openai.OpenAI:
    """Set up and return an OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        console.print("[bold red]Error:[/] OPENAI_API_KEY not found in environment variables or .env file.")
        console.print("Please set your OpenAI API key in a .env file or as an environment variable.")
        sys.exit(1)
        
    return openai.OpenAI(api_key=api_key)

def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read file: {escape(str(e))}")
        sys.exit(1)

def write_file(file_path: str, content: str) -> None:
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to write file: {escape(str(e))}")
        sys.exit(1)

def parse_frontmatter(content: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """Parse frontmatter from content using python-frontmatter.
    
    Returns:
        Tuple containing:
            - Boolean indicating if frontmatter was detected
            - Dictionary containing the frontmatter data if found, otherwise None
            - String containing the content without frontmatter if found, otherwise None
    """
    try:
        # Parse content with frontmatter
        post = frontmatter.loads(content)
        
        # Check if frontmatter was found
        if post.metadata:
            # Extract metadata and content
            metadata = dict(post.metadata)
            content_without_frontmatter = post.content
            return True, metadata, content_without_frontmatter
        else:
            # No frontmatter found
            return False, None, None
    except Exception as e:
        console.print(f"[bold yellow]Warning:[/] Failed to parse frontmatter: {escape(str(e))}")
        return False, None, None

def get_translatable_frontmatter_fields(frontmatter: Dict) -> List[str]:
    """Get a list of frontmatter fields that should be translated."""
    # Common translatable fields in various static site generators
    translatable_fields = [
        'title', 'description', 'summary', 'excerpt', 'subtitle',
        'seo_title', 'seo_description', 'meta_description',
        'abstract', 'intro', 'heading', 'subheading'
    ]
    
    # Return only fields that exist in the frontmatter
    return [field for field in translatable_fields if field in frontmatter]

def reconstruct_with_frontmatter(metadata: Dict, content: str) -> str:
    """Reconstruct content with frontmatter using python-frontmatter."""
    # Create a new post object with metadata and content
    post = frontmatter.Post(content, **metadata)
    
    # Return the serialized post
    return frontmatter.dumps(post)

def translate_frontmatter(client: openai.OpenAI, frontmatter: Dict, fields: List[str], 
                     target_language: str, model: str) -> Tuple[Dict, Dict]:
    """Translate specified fields in frontmatter.
    
    Returns:
        Tuple containing translated frontmatter and usage information
    """
    # Create a copy to avoid modifying the original
    translated_frontmatter = frontmatter.copy()
    
    if not fields:
        return translated_frontmatter, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Prepare text for translation
    fields_text = ""
    for field in fields:
        fields_text += f"{field}: {frontmatter[field]}\n\n"
    
    system_prompt = f"""You are a professional translator. Translate the following frontmatter fields to {target_language}.
Each field is in the format "field_name: content". Translate ONLY the content, not the field names.
Return the translated content in the exact same format, preserving all field names."""
    
    try:
        console.print(f"[bold]Translating frontmatter fields:[/] {', '.join(fields)}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fields_text}
            ],
        )
        
        # Extract usage statistics
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Parse the response to get translated fields
        translated_text = response.choices[0].message.content
        
        # Extract each translated field from the response
        for field in fields:
            pattern = rf"{field}: (.*?)(?:\n\n|\n$|$)"
            match = re.search(pattern, translated_text, re.DOTALL)
            if match:
                translated_value = match.group(1).strip()
                translated_frontmatter[field] = translated_value
        
        return translated_frontmatter, usage
    except Exception as e:
        console.print(f"[bold yellow]Warning:[/] Failed to translate frontmatter: {escape(str(e))}")
        # Return original frontmatter on error
        return frontmatter, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def translate_text(client: openai.OpenAI, text: str, target_language: str, model: str) -> Tuple[str, Dict]:
    """Translate text to the target language using OpenAI.
    
    Returns:
        Tuple containing translated text and usage information
    """
    
    system_prompt = f"""
    1. Read the provided text carefully, preserving all formatting, markdown, and structure exactly as they appear.
    2. Identify any block quotes and code blocks.
    3. Do not translate text in block quotes or in code blocks (including text within code blocks).
    4. Translate everything else into {target_language}.
    5. Maintain the original formatting, markdown, and structure in your output.
    6. Provide a natural-sounding translation rather than a word-for-word one.
    7. For idioms, colloquialisms, or slang, render them in an equivalent, natural way in {target_language} whenever possible.
    8. If there isn’t a direct or natural translation for a particular term or phrase, keep it in the original language and surround it with quotes if necessary.
    9. Ensure that technical terms or jargon remain accurate; if there’s no suitable translation, keep the original term.
    10. Strive for fluid, native-sounding prose that retains the tone and intent of the original text.
    """
    
    user_prompt = text
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Translating...[/]"),
            transient=True
        ) as progress:
            progress.add_task("translating", total=None)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            # Extract usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return response.choices[0].message.content, usage
    except Exception as e:
        console.print(f"[bold red]Error:[/] Translation failed: {escape(str(e))}")
        sys.exit(1)

def edit_translation(client: openai.OpenAI, translated_text: str, original_text: str, target_language: str, model: str) -> Tuple[str, Dict]:
    """Edit the translation to ensure it makes sense in the target language while preserving original meaning.
    
    Returns:
        Tuple containing edited text and usage information
    """
    
    system_prompt = f"""
    1. Carefully read the translated text alongside the original text in its entirety.
    2. Compare both texts to ensure the translation accurately reflects the original meaning.
    3. Correct any grammatical errors you find in the {target_language} text.
    4. Adjust phrasing to make it sound natural and fluent for {target_language} speakers, making sure idioms and expressions are culturally appropriate.
    5. Preserve the original tone, nuance, and style, including any formatting, markdown, and structure.
    6. Avoid adding new information or altering the core meaning.
    7. Ensure the final result doesn’t feel machine-translated but remains faithful to the source.
    8. Make only changes that genuinely improve the text's quality in {target_language}.
    7. Don't be too literal. If there isn't a direct translation, provide a natural-sounding translation.
    9. If the text contains idioms or colloquialisms, translate them into the target language while maintaining their original meaning.
    10. If the text contains technical terms or jargon, ensure that the translation is accurate and appropriate for the target audience, if there isn't a natural translation, keep it in the original language.
    11. If there is not natural translation, keep it in the original language.
"""
    
    user_prompt = f"""# ORIGINAL TEXT
{original_text}

# TRANSLATED TEXT
{translated_text}

Please review and improve the translated text to make it natural and accurate in {target_language}.
Return ONLY the improved translated text without explanations or comments."""
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Editing translation...[/]"),
            transient=True
        ) as progress:
            progress.add_task("editing", total=None)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                top_p=1.0
            )
            
            # Extract usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return response.choices[0].message.content, usage
    except Exception as e:
        console.print(f"[bold red]Error:[/] Editing failed: {escape(str(e))}")
        # Return original translation if editing fails with empty usage stats
        return translated_text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def get_language_code(language_name: str) -> str:
    """Convert a language name to ISO 639-1 two-letter code."""
    # Normalize input: lowercase and remove any non-alphanumeric characters
    language_name_normalized = re.sub(r'[^a-zA-Z0-9]', ' ', language_name.lower()).strip()
    
    # Direct lookup for common language names and variations
    direct_mappings = {
        "chinese": "zh", "mandarin": "zh", 
        "spanish": "es", "español": "es",
        "english": "en",
        "hindi": "hi",
        "arabic": "ar",
        "portuguese": "pt", "brazilian": "pt",
        "bengali": "bn",
        "russian": "ru",
        "japanese": "ja",
        "punjabi": "pa",
        "german": "de", "deutsch": "de",
        "javanese": "jv",
        "korean": "ko",
        "french": "fr", "français": "fr",
        "turkish": "tr",
        "vietnamese": "vi",
        "thai": "th",
        "italian": "it", "italiano": "it",
        "persian": "fa", "farsi": "fa",
        "polish": "pl", "polski": "pl",
        "romanian": "ro",
        "dutch": "nl",
        "greek": "el",
        "czech": "cs",
        "swedish": "sv",
        "hebrew": "he",
        "danish": "da",
        "finnish": "fi",
        "hungarian": "hu",
        "norwegian": "no"
    }
    
    # First try direct mapping
    if language_name_normalized in direct_mappings:
        return direct_mappings[language_name_normalized]
    
    # Try with pycountry
    try:
        # Try to find by name
        lang = pycountry.languages.get(name=language_name_normalized.title())
        if lang and hasattr(lang, 'alpha_2'):
            return lang.alpha_2
            
        # Try to find by partial name match
        for lang in pycountry.languages:
            if (hasattr(lang, 'name') and 
                language_name_normalized in lang.name.lower() and 
                hasattr(lang, 'alpha_2')):
                return lang.alpha_2
    except (AttributeError, KeyError):
        pass
        
    # Fall back to using the first two letters of the language name
    return language_name_normalized[:2]

def get_output_filename(input_file: str, target_language: str, output_file: Optional[str] = None) -> str:
    """Generate output filename if not provided."""
    if output_file:
        return output_file
    
    # Get language code
    language_code = get_language_code(target_language)
    
    input_path = Path(input_file)
    stem = input_path.stem
    suffix = input_path.suffix
    
    return f"{stem}.{language_code}{suffix}"

def confirm(prompt: str) -> bool:
    """Ask for user confirmation."""
    response = input(f"{prompt} (y/n): ").strip().lower()
    return response == "y" or response == "yes"

def calculate_actual_cost(usage: Dict, model: str) -> Tuple[float, str]:
    """Calculate the actual cost based on token usage."""
    if model not in MODEL_CONFIG:
        return (0.0, "Unknown model, cost could not be calculated")
    
    # Get cost per 1k tokens
    input_cost = MODEL_CONFIG[model]["input_cost"]
    output_cost = MODEL_CONFIG[model]["output_cost"]
    
    # Calculate cost
    prompt_cost = (usage["prompt_tokens"] / 1000) * input_cost
    completion_cost = (usage["completion_tokens"] / 1000) * output_cost
    total_cost = prompt_cost + completion_cost
    
    # Format cost string
    if total_cost < 0.01:
        cost_str = f"Less than $0.01"
    else:
        cost_str = f"${total_cost:.4f}"
    
    return (total_cost, cost_str)

def display_model_info() -> None:
    """Display information about available models and their costs."""
    table = Table(title="Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Max Tokens", style="green")
    table.add_column("Input Cost (per 1K tokens)", style="yellow")
    table.add_column("Output Cost (per 1K tokens)", style="yellow")
    
    for model, config in MODEL_CONFIG.items():
        table.add_row(
            model,
            f"{config['max_tokens']:,}",
            f"${config['input_cost']:.4f}",
            f"${config['output_cost']:.4f}"
        )
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Translate text files to different languages using OpenAI.")
    parser.add_argument("file", help="Path to the text file to translate")
    parser.add_argument("language", help="Target language for translation")
    parser.add_argument("-o", "--output", help="Output file path (optional)")
    parser.add_argument("-m", "--model", default="o3", help="OpenAI model to use (default: o3)")
    parser.add_argument("--no-edit", action="store_true", help="Skip the editing step (faster but may reduce quality)")
    parser.add_argument("--list-models", action="store_true", help="Display available models and pricing")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate tokens and cost, don't translate")
    
    args = parser.parse_args()
    
    # If --list-models is specified, display model info and exit
    if args.list_models:
        display_model_info()
        sys.exit(0)
    
    input_file = args.file
    target_language = args.language
    output_file = args.output
    model = args.model
    skip_edit = args.no_edit
    estimate_only = args.estimate_only
    
    if not Path(input_file).exists():
        console.print(f"[bold red]Error:[/] Input file '{escape(input_file)}' does not exist.")
        sys.exit(1)
    
    client = setup_openai_client()
    
    console.print(f"[bold]Reading file:[/] {escape(input_file)}")
    content = read_file(input_file)
    
    # Variables to track frontmatter
    has_frontmatter = False
    frontmatter_data = None
    content_without_frontmatter = None
    frontmatter_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Check if file has frontmatter (for markdown blog posts)
    if input_file.endswith(('.md', '.markdown', '.mdx')):
        has_frontmatter, frontmatter_data, content_without_frontmatter = parse_frontmatter(content)
        
        if has_frontmatter:
            console.print(f"[bold]Detected frontmatter:[/] static site generator metadata")
            # Use content without frontmatter for token count
            content_for_translation = content_without_frontmatter
            frontmatter_str = frontmatter.dumps(frontmatter.Post("", **frontmatter_data)).split('---\n\n')[0]
            frontmatter_token_count = count_tokens(frontmatter_str, model)
            console.print(f"[bold]Frontmatter size:[/] {frontmatter_token_count:,} tokens")
        else:
            content_for_translation = content
    else:
        content_for_translation = content
    
    # Check token limits
    within_limits, token_count = check_token_limits(content_for_translation, model)
    cost, cost_str = estimate_cost(token_count, model, not skip_edit)
    
    # Display token and cost information
    console.print(f"[bold]File size:[/] {len(content):,} characters, {token_count:,} tokens")
    console.print(f"[bold]Estimated cost:[/] {cost_str}")
    
    if not within_limits:
        console.print(f"[bold red]Warning:[/] This text may exceed the {model} model's token limit.")
        console.print(f"Consider splitting the file into smaller parts or using a model with a higher token limit.")
        if not estimate_only:
            if not confirm("Continue anyway?"):
                sys.exit(0)
    
    # If only estimating, exit now
    if estimate_only:
        sys.exit(0)
    
    console.print(f"[bold]Translating to:[/] {escape(target_language)}")
    console.print(f"[bold]Using model:[/] {escape(model)}")
    
    # Track total token usage
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    
    # Translate frontmatter if present
    translated_frontmatter = None
    if has_frontmatter:
        # Get fields to translate
        translatable_fields = get_translatable_frontmatter_fields(frontmatter_data)
        if translatable_fields:
            # Translate frontmatter fields
            translated_frontmatter, frontmatter_usage = translate_frontmatter(
                client, frontmatter_data, translatable_fields, target_language, model
            )
            
            # Add to total usage
            total_usage["prompt_tokens"] += frontmatter_usage["prompt_tokens"]
            total_usage["completion_tokens"] += frontmatter_usage["completion_tokens"]
            total_usage["total_tokens"] += frontmatter_usage["total_tokens"]
        else:
            translated_frontmatter = frontmatter_data
    
    # Perform translation of main content and get token usage
    translated_content, translation_usage = translate_text(client, content_for_translation, target_language, model)
    total_usage["prompt_tokens"] += translation_usage["prompt_tokens"]
    total_usage["completion_tokens"] += translation_usage["completion_tokens"]
    total_usage["total_tokens"] += translation_usage["total_tokens"]
    
    # Perform editing if not skipped
    if not skip_edit:
        console.print(f"[bold]Editing translation for fluency and accuracy...[/]")
        translated_content, edit_usage = edit_translation(client, translated_content, content_for_translation, target_language, model)
        total_usage["prompt_tokens"] += edit_usage["prompt_tokens"]
        total_usage["completion_tokens"] += edit_usage["completion_tokens"]
        total_usage["total_tokens"] += edit_usage["total_tokens"]
    
    # Reconstruct content with translated frontmatter if needed
    if has_frontmatter and translated_frontmatter:
        final_content = reconstruct_with_frontmatter(translated_frontmatter, translated_content)
    else:
        final_content = translated_content
    
    # Calculate actual cost based on token usage
    actual_cost, cost_str = calculate_actual_cost(total_usage, model)
    
    # Write output file
    output_path = get_output_filename(input_file, target_language, output_file)
    write_file(output_path, final_content)
    
    # Get the language code used for the filename
    language_code = Path(output_path).stem.split('.')[-1]
    
    # Display completion message with token usage and cost
    console.print(f"[bold green]✓[/] Translation complete!")
    console.print(f"[bold]Target language:[/] {escape(target_language)} ({language_code})")
    console.print(f"[bold]Output file:[/] {escape(output_path)}")
    
    # Display token usage table
    usage_table = Table(title="Token Usage")
    usage_table.add_column("Operation", style="cyan")
    usage_table.add_column("Input Tokens", style="green", justify="right")
    usage_table.add_column("Output Tokens", style="green", justify="right")
    usage_table.add_column("Total Tokens", style="green", justify="right")
    
    # Add frontmatter translation row if it happened
    if has_frontmatter and frontmatter_usage["total_tokens"] > 0:
        usage_table.add_row(
            "Frontmatter", 
            f"{frontmatter_usage['prompt_tokens']:,}",
            f"{frontmatter_usage['completion_tokens']:,}",
            f"{frontmatter_usage['total_tokens']:,}"
        )
    
    # Add content translation row
    usage_table.add_row(
        "Content Translation", 
        f"{translation_usage['prompt_tokens']:,}",
        f"{translation_usage['completion_tokens']:,}",
        f"{translation_usage['total_tokens']:,}"
    )
    
    # Add editing row if not skipped
    if not skip_edit:
        usage_table.add_row(
            "Content Editing", 
            f"{edit_usage['prompt_tokens']:,}",
            f"{edit_usage['completion_tokens']:,}",
            f"{edit_usage['total_tokens']:,}"
        )
    
    # Add total row
    usage_table.add_row(
        "Total", 
        f"{total_usage['prompt_tokens']:,}",
        f"{total_usage['completion_tokens']:,}",
        f"{total_usage['total_tokens']:,}",
        style="bold"
    )
    
    console.print(usage_table)
    console.print(f"[bold]Actual cost:[/] {cost_str}")

if __name__ == "__main__":
    main()
