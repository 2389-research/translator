#!/usr/bin/env python3
# ABOUTME: Command-line interface for the translator.
# ABOUTME: Handles user interaction, arguments, and displays results.

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from translator.log_interpreter import LogInterpreter
from translator.token_counter import TokenCounter
from translator.translator import Translator
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class TranslatorCLI:
    """Command-line interface for the translator."""

    @staticmethod
    def setup_openai_client() -> openai.OpenAI:
        """Set up and return an OpenAI client."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            console.print(
                "[bold red]Error:[/] OPENAI_API_KEY not found in environment variables or .env file."
            )
            console.print(
                "Please set your OpenAI API key in a .env file or as an environment variable."
            )
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
                f"${config['output_cost']:.4f}",
            )

        console.print(table)

    @staticmethod
    def display_usage_table(
        total_usage: Dict[str, int],
        translation_usage: Dict[str, int],
        edit_usage: Optional[Dict[str, int]] = None,
        frontmatter_usage: Optional[Dict[str, int]] = None,
        critique_usage: Optional[Dict[str, int]] = None,
        feedback_usage: Optional[Dict[str, int]] = None,
        critique_usages: Optional[List[Dict[str, int]]] = None,
        feedback_usages: Optional[List[Dict[str, int]]] = None,
        has_frontmatter: bool = False,
        skip_edit: bool = False,
        do_critique: bool = False,
        critique_loops: int = 1,
    ) -> None:
        """Display token usage table.

        Args:
            total_usage: Total token usage
            translation_usage: Token usage for translation
            edit_usage: Token usage for editing (optional)
            frontmatter_usage: Token usage for frontmatter translation (optional)
            critique_usage: Token usage for critique generation (optional)
            feedback_usage: Token usage for applying critique feedback (optional)
            critique_usages: List of token usages for multiple critique loops (optional)
            feedback_usages: List of token usages for multiple feedback loops (optional)
            has_frontmatter: Whether frontmatter was translated
            skip_edit: Whether editing was skipped
            do_critique: Whether critique was performed
            critique_loops: Number of critique loops performed
        """
        usage_table = Table(title="Token Usage")
        usage_table.add_column("Operation", style="cyan")
        usage_table.add_column("Input Tokens", style="green", justify="right")
        usage_table.add_column("Output Tokens", style="green", justify="right")
        usage_table.add_column("Total Tokens", style="green", justify="right")

        # Add frontmatter translation row if it happened
        if (
            has_frontmatter
            and frontmatter_usage
            and frontmatter_usage["total_tokens"] > 0
        ):
            usage_table.add_row(
                "Frontmatter",
                f"{frontmatter_usage['prompt_tokens']:,}",
                f"{frontmatter_usage['completion_tokens']:,}",
                f"{frontmatter_usage['total_tokens']:,}",
            )

        # Add content translation row
        usage_table.add_row(
            "Content Translation",
            f"{translation_usage['prompt_tokens']:,}",
            f"{translation_usage['completion_tokens']:,}",
            f"{translation_usage['total_tokens']:,}",
        )

        # Add editing row if not skipped
        if not skip_edit and edit_usage:
            usage_table.add_row(
                "Content Editing",
                f"{edit_usage['prompt_tokens']:,}",
                f"{edit_usage['completion_tokens']:,}",
                f"{edit_usage['total_tokens']:,}",
            )

        # Add critique and feedback rows if performed
        if do_critique:
            # If we have multiple critique loops, show each one
            if critique_usages and feedback_usages and len(critique_usages) > 0:
                for i, (crit_usage, feed_usage) in enumerate(
                    zip(critique_usages, feedback_usages)
                ):
                    if crit_usage and crit_usage["total_tokens"] > 0:
                        usage_table.add_row(
                            f"Critique Generation (Loop {i+1})",
                            f"{crit_usage['prompt_tokens']:,}",
                            f"{crit_usage['completion_tokens']:,}",
                            f"{crit_usage['total_tokens']:,}",
                        )
                    if feed_usage and feed_usage["total_tokens"] > 0:
                        usage_table.add_row(
                            f"Critique Application (Loop {i+1})",
                            f"{feed_usage['prompt_tokens']:,}",
                            f"{feed_usage['completion_tokens']:,}",
                            f"{feed_usage['total_tokens']:,}",
                        )
            # Fallback to original behavior for backward compatibility
            elif critique_usage and critique_usage["total_tokens"] > 0:
                usage_table.add_row(
                    "Critique Generation",
                    f"{critique_usage['prompt_tokens']:,}",
                    f"{critique_usage['completion_tokens']:,}",
                    f"{critique_usage['total_tokens']:,}",
                )
                if feedback_usage and feedback_usage["total_tokens"] > 0:
                    usage_table.add_row(
                        "Critique Application",
                        f"{feedback_usage['prompt_tokens']:,}",
                        f"{feedback_usage['completion_tokens']:,}",
                        f"{feedback_usage['total_tokens']:,}",
                    )

        # Add total row
        usage_table.add_row(
            "Total",
            f"{total_usage['prompt_tokens']:,}",
            f"{total_usage['completion_tokens']:,}",
            f"{total_usage['total_tokens']:,}",
            style="bold",
        )

        console.print(usage_table)

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Parse command-line arguments.

        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Translate text files to different languages using OpenAI."
        )
        parser.add_argument("file", help="Path to the text file to translate")
        parser.add_argument("language", help="Target language for translation")
        parser.add_argument("-o", "--output", help="Output file path (optional)")
        parser.add_argument(
            "-m", "--model", default="o3", help="OpenAI model to use (default: o3)"
        )
        parser.add_argument(
            "--no-edit",
            action="store_true",
            help="Skip the editing step (faster but may reduce quality)",
        )
        parser.add_argument(
            "--no-critique",
            action="store_true",
            help="Skip the aggressive critique step (faster but may reduce quality)",
        )
        parser.add_argument(
            "--critique-loops",
            type=int,
            default=4,
            help="Number of critique-revision loops to perform (default: 4, max: 5)",
        )
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="Display available models and pricing",
        )
        parser.add_argument(
            "--estimate-only",
            action="store_true",
            help="Only estimate tokens and cost, don't translate",
        )

        return parser.parse_args()

    @classmethod
    def _parse_and_validate_args(
        cls, args: argparse.Namespace
    ) -> Tuple[str, str, Optional[str], str, bool, bool, int, bool, bool]:
        """Parse and validate command line arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            Tuple containing: input_file, target_language, output_file, model,
            skip_edit, do_critique, critique_loops, estimate_only, has_valid_input
        """
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
        critique_loops = args.critique_loops if do_critique else 0
        # Limit critique loops to a reasonable number
        critique_loops = min(max(critique_loops, 0), 5)
        estimate_only = args.estimate_only

        # Validate input file
        has_valid_input = True
        if not Path(input_file).exists():
            console.print(
                f"[bold red]Error:[/] Input file '{escape(input_file)}' does not exist."
            )
            has_valid_input = False

        return (
            input_file,
            target_language,
            output_file,
            model,
            skip_edit,
            do_critique,
            critique_loops,
            estimate_only,
            has_valid_input,
        )

    @classmethod
    def _process_content(
        cls, input_file: str, content: str, model: str
    ) -> Tuple[str, bool, Optional[Dict], Dict, str]:
        """Process input content and extract frontmatter if present.

        Args:
            input_file: Path to the input file
            content: Content of the input file
            model: Model to use for translation

        Returns:
            Tuple containing: content_for_translation, has_frontmatter,
            frontmatter_data, frontmatter_usage, content_size_message
        """
        # Variables to track frontmatter
        has_frontmatter = False
        frontmatter_data = None
        content_without_frontmatter = None
        frontmatter_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Check if file has frontmatter (for markdown blog posts)
        if input_file.endswith((".md", ".markdown", ".mdx")):
            has_frontmatter, frontmatter_data, content_without_frontmatter = (
                FrontmatterHandler.parse_frontmatter(content)
            )

            if has_frontmatter:
                console.print(
                    "[bold]Detected frontmatter:[/] static site generator metadata"
                )
                # Use content without frontmatter for token count
                content_for_translation = content_without_frontmatter
                frontmatter_str = frontmatter.dumps(
                    frontmatter.Post("", **frontmatter_data)
                ).split("---\n\n")[0]
                frontmatter_token_count = TokenCounter.count_tokens(
                    frontmatter_str, model
                )
                content_size_message = (
                    f"[bold]Frontmatter size:[/] {frontmatter_token_count:,} tokens"
                )
            else:
                content_for_translation = content
                content_size_message = ""
        else:
            content_for_translation = content
            content_size_message = ""

        return (
            content_for_translation,
            has_frontmatter,
            frontmatter_data,
            frontmatter_usage,
            content_size_message,
        )

    @classmethod
    def _check_limits_and_estimate_cost(
        cls,
        content: str,
        content_for_translation: str,
        model: str,
        skip_edit: bool,
        do_critique: bool,
        critique_loops: int,
        estimate_only: bool,
    ) -> Tuple[bool, int, float, str, bool]:
        """Check token limits and estimate cost.

        Args:
            content: Full content of the input file
            content_for_translation: Content ready for translation (without frontmatter if any)
            model: Model to use for translation
            skip_edit: Whether to skip the editing step
            do_critique: Whether to perform critique
            critique_loops: Number of critique loops to perform
            estimate_only: Whether to only estimate tokens and cost

        Returns:
            Tuple containing: within_limits, token_count, cost, cost_str, should_continue
        """
        # Check token limits
        within_limits, token_count = TokenCounter.check_token_limits(
            content_for_translation,
            model,
            with_edit=not skip_edit,
            with_critique=do_critique,
            critique_loops=critique_loops,
        )
        cost, cost_str = CostEstimator.estimate_cost(
            token_count, model, not skip_edit, do_critique, critique_loops
        )

        # Display token and cost information
        console.print(
            f"[bold]File size:[/] {len(content):,} characters, {token_count:,} tokens"
        )
        console.print(f"[bold]Estimated cost:[/] {cost_str}")

        should_continue = True
        if not within_limits:
            console.print(
                f"[bold red]Warning:[/] This text may exceed the {model} model's token limit."
            )
            console.print(
                "Consider splitting the file into smaller parts or using a model with a higher token limit."
            )
            if not estimate_only:
                if not cls.confirm("Continue anyway?"):
                    should_continue = False

        return within_limits, token_count, cost, cost_str, should_continue

    @classmethod
    def _translate_frontmatter(
        cls,
        has_frontmatter: bool,
        frontmatter_data: Optional[Dict],
        translator: Translator,
        target_language: str,
        model: str,
    ) -> Tuple[Optional[Dict], Dict, Dict]:
        """Translate frontmatter if present.

        Args:
            has_frontmatter: Whether the input has frontmatter
            frontmatter_data: Frontmatter data if present
            translator: Translator instance
            target_language: Target language for translation
            model: Model to use for translation

        Returns:
            Tuple containing: translated_frontmatter, frontmatter_usage, total_usage
        """
        # Initialize usage tracking
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        frontmatter_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Translate frontmatter if present
        translated_frontmatter = None
        if has_frontmatter and frontmatter_data:
            # Get fields to translate
            translatable_fields = (
                FrontmatterHandler.get_translatable_frontmatter_fields(frontmatter_data)
            )
            if translatable_fields:
                # Display UI message
                console.print(
                    f"[bold]Translating frontmatter fields:[/] {', '.join(translatable_fields)}"
                )
                
                # Translate frontmatter fields
                translated_frontmatter, frontmatter_usage, error_msg = (
                    translator.translate_frontmatter(
                        frontmatter_data, translatable_fields, target_language, model
                    )
                )
                
                # Handle any error
                if error_msg:
                    console.print(f"[bold yellow]Warning:[/] {error_msg}")

                # Add to total usage
                total_usage["prompt_tokens"] += frontmatter_usage["prompt_tokens"]
                total_usage["completion_tokens"] += frontmatter_usage[
                    "completion_tokens"
                ]
                total_usage["total_tokens"] += frontmatter_usage["total_tokens"]
            else:
                translated_frontmatter = frontmatter_data

        return translated_frontmatter, frontmatter_usage, total_usage

    @classmethod
    def _translate_content(
        cls,
        content_for_translation: str,
        translator: Translator,
        target_language: str,
        model: str,
        total_usage: Dict,
    ) -> Tuple[str, Dict]:
        """Translate main content and update usage tracking.

        Args:
            content_for_translation: Content to translate
            translator: Translator instance
            target_language: Target language for translation
            model: Model to use for translation
            total_usage: Current token usage to update

        Returns:
            Tuple containing: translated_content, translation_usage
        """
        # Use Rich Progress bar for translation
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Translating...[/]"),
            transient=True,
        ) as progress:
            progress.add_task("translating", total=None)
            
            # Perform translation of main content and get token usage
            translated_content, translation_usage, error_msg = translator.translate_text(
                content_for_translation, target_language, model
            )
        
        # Handle any error
        if error_msg:
            console.print(f"[bold red]Error:[/] {error_msg}")
            sys.exit(1)
            
        # Update usage tracking
        total_usage["prompt_tokens"] += translation_usage["prompt_tokens"]
        total_usage["completion_tokens"] += translation_usage["completion_tokens"]
        total_usage["total_tokens"] += translation_usage["total_tokens"]

        return translated_content, translation_usage

    @classmethod
    def _edit_content(
        cls,
        skip_edit: bool,
        translated_content: str,
        content_for_translation: str,
        translator: Translator,
        target_language: str,
        model: str,
        total_usage: Dict,
    ) -> Tuple[str, Dict]:
        """Edit translated content if not skipped and update usage tracking.

        Args:
            skip_edit: Whether to skip the editing step
            translated_content: Translated content to edit
            content_for_translation: Original content
            translator: Translator instance
            target_language: Target language for translation
            model: Model to use for translation
            total_usage: Current token usage to update

        Returns:
            Tuple containing: edited_content, edit_usage
        """
        # Initialize edit usage tracking
        edit_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Perform editing if not skipped
        if not skip_edit:
            console.print("[bold]Editing translation for fluency and accuracy...[/]")
            
            # Use Rich Progress bar for editing
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Editing translation...[/]"),
                transient=True,
            ) as progress:
                progress.add_task("editing", total=None)
                
                # Perform editing
                translated_content, edit_usage, error_msg = translator.edit_translation(
                    translated_content, content_for_translation, target_language, model
                )
            
            # Handle any error
            if error_msg:
                console.print(f"[bold yellow]Warning:[/] {error_msg}")
                
            # Update usage tracking
            total_usage["prompt_tokens"] += edit_usage["prompt_tokens"]
            total_usage["completion_tokens"] += edit_usage["completion_tokens"]
            total_usage["total_tokens"] += edit_usage["total_tokens"]

        return translated_content, edit_usage

    @classmethod
    def _perform_critique_loops(
        cls,
        do_critique: bool,
        critique_loops: int,
        translated_content: str,
        content_for_translation: str,
        translator: Translator,
        target_language: str,
        model: str,
        total_usage: Dict,
    ) -> Tuple[str, Dict, Dict, List[Dict], List[Dict]]:
        """Perform critique loops if requested.

        Args:
            do_critique: Whether to perform critique
            critique_loops: Number of critique loops to perform
            translated_content: Translated content to critique
            content_for_translation: Original content
            translator: Translator instance
            target_language: Target language for translation
            model: Model to use for translation
            total_usage: Current token usage to update

        Returns:
            Tuple containing: improved_content, critique_usage, feedback_usage,
            critique_usages, feedback_usages
        """
        # Initialize critique usage tracking
        critique_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        feedback_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Lists to track usage across multiple critique loops
        critique_usages = []
        feedback_usages = []

        if do_critique and critique_loops > 0:
            # Store all critiques for logging
            all_critiques = []

            for loop in range(critique_loops):
                # Generate critique for the current version
                console.print(
                    f"[bold]Critique loop {loop+1}/{critique_loops}: Generating critique...[/]"
                )
                
                # Use Rich Progress bar for critique generation
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold red]Generating critique...[/]"),
                    transient=True,
                ) as progress:
                    progress.add_task("critiquing", total=None)
                    
                    # Generate critique
                    _, loop_critique_usage, critique_feedback, error_msg = (
                        translator.critique_translation(
                            translated_content,
                            content_for_translation,
                            target_language,
                            model,
                        )
                    )
                
                # Handle any critique error
                if error_msg:
                    console.print(f"[bold yellow]Warning:[/] {error_msg}")
                    continue

                # Add usage to the running total
                total_usage["prompt_tokens"] += loop_critique_usage["prompt_tokens"]
                total_usage["completion_tokens"] += loop_critique_usage[
                    "completion_tokens"
                ]
                total_usage["total_tokens"] += loop_critique_usage["total_tokens"]

                # Store for reporting
                critique_usages.append(loop_critique_usage)
                all_critiques.append(critique_feedback)

                # Show critique summary
                critique_lines = critique_feedback.split("\n")
                preview_lines = 10  # Show first 10 lines as a preview
                console.print(f"[bold yellow]Critique summary (loop {loop+1}):[/]")
                for i, line in enumerate(critique_lines[:preview_lines]):
                    console.print(f"  {line}")
                if len(critique_lines) > preview_lines:
                    console.print(
                        f"  [dim]...and {len(critique_lines) - preview_lines} more lines[/dim]"
                    )

                # Apply critique feedback
                console.print(
                    f"[bold]Critique loop {loop+1}/{critique_loops}: Applying critique feedback...[/]"
                )
                
                # Use Rich Progress bar for applying feedback
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Applying critique feedback...[/]"),
                    transient=True,
                ) as progress:
                    progress.add_task("applying_feedback", total=None)
                    
                    # Apply feedback
                    translated_content, loop_feedback_usage, error_msg = (
                        translator.apply_critique_feedback(
                            translated_content,
                            content_for_translation,
                            critique_feedback,
                            target_language,
                            model,
                        )
                    )
                
                # Handle any feedback application error
                if error_msg:
                    console.print(f"[bold yellow]Warning:[/] {error_msg}")

                # Add usage to the running total
                total_usage["prompt_tokens"] += loop_feedback_usage["prompt_tokens"]
                total_usage["completion_tokens"] += loop_feedback_usage[
                    "completion_tokens"
                ]
                total_usage["total_tokens"] += loop_feedback_usage["total_tokens"]

                # Store for reporting
                feedback_usages.append(loop_feedback_usage)

            # For compatibility with existing code, store the last loop's usage in the original variables
            critique_usage = (
                critique_usages[-1]
                if critique_usages
                else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            feedback_usage = (
                feedback_usages[-1]
                if feedback_usages
                else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

            # Store all critiques in translator's log for logging
            translator.translation_log["all_critiques"] = all_critiques

        return (
            translated_content,
            critique_usage,
            feedback_usage,
            critique_usages,
            feedback_usages,
        )

    @classmethod
    def _finalize_and_save(
        cls,
        has_frontmatter: bool,
        translated_frontmatter: Optional[Dict],
        translated_content: str,
        input_file: str,
        target_language: str,
        output_file: Optional[str],
        total_usage: Dict,
        model: str,
        translator: Translator,
        skip_edit: bool,
        do_critique: bool,
        critique_loops: int,
        translation_usage: Dict,
        edit_usage: Dict,
        frontmatter_usage: Dict,
        critique_usage: Dict,
        feedback_usage: Dict,
        critique_usages: List[Dict],
        feedback_usages: List[Dict],
    ) -> None:
        """Finalize translation, save results, and display summary information.

        Args:
            has_frontmatter: Whether the input has frontmatter
            translated_frontmatter: Translated frontmatter data if present
            translated_content: Translated content
            input_file: Path to the input file
            target_language: Target language for translation
            output_file: Optional custom output file path
            total_usage: Total token usage
            model: Model used for translation
            translator: Translator instance
            skip_edit: Whether editing was skipped
            do_critique: Whether critique was performed
            critique_loops: Number of critique loops performed
            translation_usage: Token usage for translation
            edit_usage: Token usage for editing
            frontmatter_usage: Token usage for frontmatter translation
            critique_usage: Token usage for critique generation
            feedback_usage: Token usage for applying critique feedback
            critique_usages: List of token usages for multiple critique loops
            feedback_usages: List of token usages for multiple feedback loops
        """
        # Reconstruct content with translated frontmatter if needed
        if has_frontmatter and translated_frontmatter:
            final_content = FrontmatterHandler.reconstruct_with_frontmatter(
                translated_frontmatter, translated_content
            )
        else:
            final_content = translated_content

        # Calculate actual cost based on token usage
        actual_cost, cost_str = CostEstimator.calculate_actual_cost(total_usage, model)

        # Write output file
        output_path = FileHandler.get_output_filename(
            input_file, target_language, output_file
        )
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
            "critique_loops": critique_loops,
            "has_frontmatter": has_frontmatter,
            "token_usage": total_usage,
            "cost": cost_str,
            "prompts_and_responses": translator.translation_log,
            # Include more detailed info for multiple critique loops
            "critique_loop_details": {
                "critique_usages": critique_usages,
                "feedback_usages": feedback_usages,
            },
        }
        FileHandler.write_log(log_path, log_data)

        # Display completion message with token usage and cost
        console.print("[bold green]✓[/] Translation complete!")
        console.print(
            f"[bold]Target language:[/] {escape(target_language)} ({language_code})"
        )
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
            critique_usages=critique_usages,
            feedback_usages=feedback_usages,
            has_frontmatter=has_frontmatter,
            skip_edit=skip_edit,
            do_critique=do_critique,
            critique_loops=critique_loops,
        )

        console.print(f"[bold]Actual cost:[/] {cost_str}")

        # Generate narrative interpretation
        cls._generate_narrative(
            client=translator.client, log_data=log_data, output_path=output_path
        )

    @classmethod
    def _generate_narrative(
        cls, client: openai.OpenAI, log_data: Dict, output_path: str
    ) -> None:
        """Generate and save a narrative interpretation of the translation process.

        Args:
            client: OpenAI client instance
            log_data: Translation log data
            output_path: Path to the output file
        """
        # Automatically generate a narrative interpretation of the log
        console.print(
            "[bold]Generating narrative interpretation of the translation process...[/]"
        )
        log_interpreter = LogInterpreter(client)

        # Generate the narrative
        narrative = log_interpreter.generate_narrative(log_data, "o4-mini")

        # Get the narrative filename - format: filename.languagecode.log
        base_parts = Path(output_path).stem.split(".")
        if len(base_parts) >= 2:
            # This handles the standard output format: filename.languagecode.ext
            base = ".".join(base_parts[:2])  # Take filename and language code
        else:
            # Fallback for custom output paths
            base = Path(output_path).stem

        narrative_path = str(Path(output_path).parent / f"{base}.log")

        # Write the narrative to a file
        log_interpreter.write_narrative(narrative_path, narrative)

        console.print("[bold green]✓[/] Narrative interpretation generated!")
        console.print(f"[bold]Narrative file:[/] {escape(narrative_path)}")

        # Print a preview of the narrative
        console.print("\n[bold]Narrative interpretation preview:[/]")
        preview_lines = narrative.split("\n")[:5]  # First 5 lines
        for line in preview_lines:
            console.print(f"  {line}")
        if len(narrative.split("\n")) > 5:
            console.print("  [dim]...(see full narrative in the narrative file)[/dim]")

    @classmethod
    def translate_file(
        cls,
        input_file: str,
        target_language: str,
        output_file: Optional[str],
        model: str,
        skip_edit: bool,
        do_critique: bool,
        critique_loops: int,
        translator: Translator,
    ) -> Tuple[str, str, str]:
        """Translate a file to the target language.

        Args:
            input_file: Path to the input file
            target_language: Target language for translation
            output_file: Optional custom output file path
            model: Model to use for translation
            skip_edit: Whether to skip the editing step
            do_critique: Whether to perform critique
            critique_loops: Number of critique loops to perform
            translator: Translator instance

        Returns:
            Tuple containing: output_path, log_path, narrative_path
        """
        console.print(f"[bold]Reading file:[/] {escape(input_file)}")
        content = FileHandler.read_file(input_file)

        # Process content and extract frontmatter if present
        (
            content_for_translation,
            has_frontmatter,
            frontmatter_data,
            frontmatter_usage,
            content_size_message,
        ) = cls._process_content(input_file, content, model)
        if content_size_message:
            console.print(content_size_message)

        console.print(f"[bold]Translating to:[/] {escape(target_language)}")
        console.print(f"[bold]Using model:[/] {escape(model)}")

        # Translate frontmatter if present
        translated_frontmatter, frontmatter_usage, total_usage = (
            cls._translate_frontmatter(
                has_frontmatter, frontmatter_data, translator, target_language, model
            )
        )

        # Translate main content
        translated_content, translation_usage = cls._translate_content(
            content_for_translation, translator, target_language, model, total_usage
        )

        # Edit content if not skipped
        translated_content, edit_usage = cls._edit_content(
            skip_edit,
            translated_content,
            content_for_translation,
            translator,
            target_language,
            model,
            total_usage,
        )

        # Perform critique loops if requested
        (
            translated_content,
            critique_usage,
            feedback_usage,
            critique_usages,
            feedback_usages,
        ) = cls._perform_critique_loops(
            do_critique,
            critique_loops,
            translated_content,
            content_for_translation,
            translator,
            target_language,
            model,
            total_usage,
        )

        # Calculate output paths before finalizing
        output_path = FileHandler.get_output_filename(
            input_file, target_language, output_file
        )
        log_path = FileHandler.get_log_filename(output_path)

        # Get the narrative filename - format: filename.languagecode.log
        base_parts = Path(output_path).stem.split(".")
        if len(base_parts) >= 2:
            # This handles the standard output format: filename.languagecode.ext
            base = ".".join(base_parts[:2])  # Take filename and language code
        else:
            # Fallback for custom output paths
            base = Path(output_path).stem

        narrative_path = str(Path(output_path).parent / f"{base}.log")

        # Finalize translation, save results, and display summary
        cls._finalize_and_save(
            has_frontmatter,
            translated_frontmatter,
            translated_content,
            input_file,
            target_language,
            output_file,
            total_usage,
            model,
            translator,
            skip_edit,
            do_critique,
            critique_loops,
            translation_usage,
            edit_usage,
            frontmatter_usage,
            critique_usage,
            feedback_usage,
            critique_usages,
            feedback_usages,
        )

        return output_path, log_path, narrative_path

    @classmethod
    def run(cls) -> None:
        """Run the translator command-line interface."""
        args = cls.parse_arguments()

        # Parse and validate arguments
        (
            input_file,
            target_language,
            output_file,
            model,
            skip_edit,
            do_critique,
            critique_loops,
            estimate_only,
            has_valid_input,
        ) = cls._parse_and_validate_args(args)

        if not has_valid_input:
            sys.exit(1)

        # Read input file
        content = FileHandler.read_file(input_file)

        # Process content and extract frontmatter if present
        (
            content_for_translation,
            has_frontmatter,
            frontmatter_data,
            frontmatter_usage,
            content_size_message,
        ) = cls._process_content(input_file, content, model)
        if content_size_message:
            console.print(content_size_message)

        # Check token limits and estimate cost
        within_limits, token_count, cost, cost_str, should_continue = (
            cls._check_limits_and_estimate_cost(
                content,
                content_for_translation,
                model,
                skip_edit,
                do_critique,
                critique_loops,
                estimate_only,
            )
        )

        if not should_continue:
            sys.exit(0)

        # If only estimating, exit now
        if estimate_only:
            sys.exit(0)

        # Set up OpenAI client and translator
        client = cls.setup_openai_client()
        translator = Translator(client)

        # Translate the file
        cls.translate_file(
            input_file,
            target_language,
            output_file,
            model,
            skip_edit,
            do_critique,
            critique_loops,
            translator,
        )
