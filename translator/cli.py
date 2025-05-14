#!/usr/bin/env python3
# ABOUTME: Command-line interface for the translator.
# ABOUTME: Handles user interaction, arguments, and displays results.

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import frontmatter
import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from translator.config import ModelConfig
from translator.cost import CostEstimator
from translator.file_handler import FileHandler
from translator.frontmatter_handler import FrontmatterHandler
from translator.token_counter import TokenCounter
from translator.translator import Translator

console = Console()


class TranslatorCLI:
    """Command-line interface for the translator."""

    @staticmethod
    def setup_openai_client() -> openai.OpenAI:
        """Set up and return an OpenAI client."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            console.print("[bold red]Error:[/] OPENAI_API_KEY not found in environment variables or .env file.")
            console.print("Please set your OpenAI API key in a .env file or as an environment variable.")
            sys.exit(1)
            
        return openai.OpenAI(api_key=api_key)

    @staticmethod
    def confirm(prompt: str) -> bool:
        """Ask for user confirmation."""
        response = input(f"{prompt} (y/n): ").strip().lower()
        return response == "y" or response == "yes"

    @staticmethod
    def display_model_info() -> None:
        """Display information about available models and their costs."""
        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Max Tokens", style="green")
        table.add_column("Input Cost (per 1K tokens)", style="yellow")
        table.add_column("Output Cost (per 1K tokens)", style="yellow")
        
        for model, config in ModelConfig.MODELS.items():
            table.add_row(
                model,
                f"{config['max_tokens']:,}",
                f"${config['input_cost']:.4f}",
                f"${config['output_cost']:.4f}"
            )
        
        console.print(table)

    @staticmethod
    def display_usage_table(total_usage: Dict[str, int], translation_usage: Dict[str, int],
                          edit_usage: Optional[Dict[str, int]] = None, 
                          frontmatter_usage: Optional[Dict[str, int]] = None,
                          critique_usage: Optional[Dict[str, int]] = None,
                          feedback_usage: Optional[Dict[str, int]] = None,
                          has_frontmatter: bool = False,
                          skip_edit: bool = False, 
                          do_critique: bool = False) -> None:
        """Display token usage table.
        
        Args:
            total_usage: Total token usage
            translation_usage: Token usage for translation
            edit_usage: Token usage for editing (optional)
            frontmatter_usage: Token usage for frontmatter translation (optional)
            critique_usage: Token usage for critique generation (optional)
            feedback_usage: Token usage for applying critique feedback (optional)
            has_frontmatter: Whether frontmatter was translated
            skip_edit: Whether editing was skipped
            do_critique: Whether critique was performed
        """
        usage_table = Table(title="Token Usage")
        usage_table.add_column("Operation", style="cyan")
        usage_table.add_column("Input Tokens", style="green", justify="right")
        usage_table.add_column("Output Tokens", style="green", justify="right")
        usage_table.add_column("Total Tokens", style="green", justify="right")
        
        # Add frontmatter translation row if it happened
        if has_frontmatter and frontmatter_usage and frontmatter_usage["total_tokens"] > 0:
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
        if not skip_edit and edit_usage:
            usage_table.add_row(
                "Content Editing", 
                f"{edit_usage['prompt_tokens']:,}",
                f"{edit_usage['completion_tokens']:,}",
                f"{edit_usage['total_tokens']:,}"
            )
        
        # Add critique and feedback rows if performed
        if do_critique:
            if critique_usage and critique_usage["total_tokens"] > 0:
                usage_table.add_row(
                    "Critique Generation", 
                    f"{critique_usage['prompt_tokens']:,}",
                    f"{critique_usage['completion_tokens']:,}",
                    f"{critique_usage['total_tokens']:,}"
                )
            if feedback_usage and feedback_usage["total_tokens"] > 0:
                usage_table.add_row(
                    "Critique Application", 
                    f"{feedback_usage['prompt_tokens']:,}",
                    f"{feedback_usage['completion_tokens']:,}",
                    f"{feedback_usage['total_tokens']:,}"
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

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Parse command-line arguments.
        
        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(description="Translate text files to different languages using OpenAI.")
        parser.add_argument("file", help="Path to the text file to translate")
        parser.add_argument("language", help="Target language for translation")
        parser.add_argument("-o", "--output", help="Output file path (optional)")
        parser.add_argument("-m", "--model", default="o3", help="OpenAI model to use (default: o3)")
        parser.add_argument("--no-edit", action="store_true", help="Skip the editing step (faster but may reduce quality)")
        parser.add_argument("--no-critique", action="store_true", help="Skip the aggressive critique step (faster but may reduce quality)")
        parser.add_argument("--list-models", action="store_true", help="Display available models and pricing")
        parser.add_argument("--estimate-only", action="store_true", help="Only estimate tokens and cost, don't translate")
        
        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        """Run the translator command-line interface."""
        args = cls.parse_arguments()
        
        # If --list-models is specified, display model info and exit
        if args.list_models:
            cls.display_model_info()
            sys.exit(0)
        
        input_file = args.file
        target_language = args.language
        output_file = args.output
        model = args.model
        skip_edit = args.no_edit
        do_critique = not args.no_critique  # Critique is enabled by default
        estimate_only = args.estimate_only
        
        if not Path(input_file).exists():
            console.print(f"[bold red]Error:[/] Input file '{escape(input_file)}' does not exist.")
            sys.exit(1)
        
        client = cls.setup_openai_client()
        translator = Translator(client)
        
        console.print(f"[bold]Reading file:[/] {escape(input_file)}")
        content = FileHandler.read_file(input_file)
        
        # Variables to track frontmatter
        has_frontmatter = False
        frontmatter_data = None
        content_without_frontmatter = None
        frontmatter_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Check if file has frontmatter (for markdown blog posts)
        if input_file.endswith(('.md', '.markdown', '.mdx')):
            has_frontmatter, frontmatter_data, content_without_frontmatter = FrontmatterHandler.parse_frontmatter(content)
            
            if has_frontmatter:
                console.print(f"[bold]Detected frontmatter:[/] static site generator metadata")
                # Use content without frontmatter for token count
                content_for_translation = content_without_frontmatter
                frontmatter_str = frontmatter.dumps(frontmatter.Post("", **frontmatter_data)).split('---\n\n')[0]
                frontmatter_token_count = TokenCounter.count_tokens(frontmatter_str, model)
                console.print(f"[bold]Frontmatter size:[/] {frontmatter_token_count:,} tokens")
            else:
                content_for_translation = content
        else:
            content_for_translation = content
        
        # Check token limits
        within_limits, token_count = TokenCounter.check_token_limits(content_for_translation, model)
        cost, cost_str = CostEstimator.estimate_cost(token_count, model, not skip_edit, do_critique)
        
        # Display token and cost information
        console.print(f"[bold]File size:[/] {len(content):,} characters, {token_count:,} tokens")
        console.print(f"[bold]Estimated cost:[/] {cost_str}")
        
        if not within_limits:
            console.print(f"[bold red]Warning:[/] This text may exceed the {model} model's token limit.")
            console.print(f"Consider splitting the file into smaller parts or using a model with a higher token limit.")
            if not estimate_only:
                if not cls.confirm("Continue anyway?"):
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
            translatable_fields = FrontmatterHandler.get_translatable_frontmatter_fields(frontmatter_data)
            if translatable_fields:
                # Translate frontmatter fields
                translated_frontmatter, frontmatter_usage = translator.translate_frontmatter(
                    frontmatter_data, translatable_fields, target_language, model
                )
                
                # Add to total usage
                total_usage["prompt_tokens"] += frontmatter_usage["prompt_tokens"]
                total_usage["completion_tokens"] += frontmatter_usage["completion_tokens"]
                total_usage["total_tokens"] += frontmatter_usage["total_tokens"]
            else:
                translated_frontmatter = frontmatter_data
        
        # Perform translation of main content and get token usage
        translated_content, translation_usage = translator.translate_text(content_for_translation, target_language, model)
        total_usage["prompt_tokens"] += translation_usage["prompt_tokens"]
        total_usage["completion_tokens"] += translation_usage["completion_tokens"]
        total_usage["total_tokens"] += translation_usage["total_tokens"]
        
        # Perform editing if not skipped
        edit_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not skip_edit:
            console.print(f"[bold]Editing translation for fluency and accuracy...[/]")
            translated_content, edit_usage = translator.edit_translation(translated_content, content_for_translation, target_language, model)
            total_usage["prompt_tokens"] += edit_usage["prompt_tokens"]
            total_usage["completion_tokens"] += edit_usage["completion_tokens"]
            total_usage["total_tokens"] += edit_usage["total_tokens"]
        
        # Perform critique step if requested
        critique_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        feedback_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if do_critique:
            # Generate critique
            console.print(f"[bold]Generating critique of translation...[/]")
            _, critique_usage, critique_feedback = translator.critique_translation(
                translated_content, content_for_translation, target_language, model
            )
            total_usage["prompt_tokens"] += critique_usage["prompt_tokens"]
            total_usage["completion_tokens"] += critique_usage["completion_tokens"]
            total_usage["total_tokens"] += critique_usage["total_tokens"]
            
            # Show critique summary
            critique_lines = critique_feedback.split('\n')
            preview_lines = 10  # Show first 10 lines as a preview
            console.print("[bold yellow]Critique summary:[/]")
            for i, line in enumerate(critique_lines[:preview_lines]):
                console.print(f"  {line}")
            if len(critique_lines) > preview_lines:
                console.print(f"  [dim]...and {len(critique_lines) - preview_lines} more lines[/dim]")
            
            # Apply critique feedback
            console.print(f"[bold]Applying critique feedback to improve translation...[/]")
            translated_content, feedback_usage = translator.apply_critique_feedback(
                translated_content, content_for_translation, critique_feedback, target_language, model
            )
            total_usage["prompt_tokens"] += feedback_usage["prompt_tokens"]
            total_usage["completion_tokens"] += feedback_usage["completion_tokens"]
            total_usage["total_tokens"] += feedback_usage["total_tokens"]
        
        # Reconstruct content with translated frontmatter if needed
        if has_frontmatter and translated_frontmatter:
            final_content = FrontmatterHandler.reconstruct_with_frontmatter(translated_frontmatter, translated_content)
        else:
            final_content = translated_content
        
        # Calculate actual cost based on token usage
        actual_cost, cost_str = CostEstimator.calculate_actual_cost(total_usage, model)
        
        # Write output file
        output_path = FileHandler.get_output_filename(input_file, target_language, output_file)
        FileHandler.write_file(output_path, final_content)
        
        # Get the language code used for the filename
        from translator.language import LanguageHandler
        language_code = LanguageHandler.get_language_code(target_language)
        
        # Write translation log to a file
        log_path = FileHandler.get_log_filename(output_path)
        log_data = {
            "input_file": input_file,
            "output_file": output_path,
            "target_language": target_language,
            "language_code": language_code,
            "model": model,
            "skip_edit": skip_edit,
            "do_critique": do_critique,
            "has_frontmatter": has_frontmatter,
            "token_usage": total_usage,
            "cost": cost_str,
            "prompts_and_responses": translator.translation_log
        }
        FileHandler.write_log(log_path, log_data)
        
        # Display completion message with token usage and cost
        console.print(f"[bold green]✓[/] Translation complete!")
        console.print(f"[bold]Target language:[/] {escape(target_language)} ({language_code})")
        console.print(f"[bold]Output file:[/] {escape(output_path)}")
        console.print(f"[bold]Log file:[/] {escape(log_path)}")
        
        # Display token usage table
        cls.display_usage_table(
            total_usage=total_usage,
            translation_usage=translation_usage,
            edit_usage=edit_usage,
            frontmatter_usage=frontmatter_usage,
            critique_usage=critique_usage,
            feedback_usage=feedback_usage,
            has_frontmatter=has_frontmatter,
            skip_edit=skip_edit,
            do_critique=do_critique
        )
        
        console.print(f"[bold]Actual cost:[/] {cost_str}")