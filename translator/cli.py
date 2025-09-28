#!/usr/bin/env python3
# ABOUTME: Command-line interface for the translator.
# ABOUTME: Handles user interaction, arguments, and displays results.

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import frontmatter
import openai
import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from translator.config import ModelConfig
from translator.cost import CostEstimator
from translator.file_handler import FileHandler
from translator.frontmatter_handler import FrontmatterHandler
from translator.log_interpreter import LogInterpreter
from translator.token_counter import TokenCounter
from translator.translator import Translator
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Cancellation handler for early termination of streaming completions
class CancellationHandler:
    """Handles cancellation requests for streaming completions.
    
    This class provides a mechanism to handle keyboard interrupts (Ctrl+C) during
    streaming completions. It registers a signal handler that sets a flag when
    the user presses Ctrl+C, allowing the application to clean up gracefully.
    
    A second Ctrl+C will force an immediate exit.
    """
    
    def __init__(self):
        """Initialize the cancellation handler."""
        self.cancel_requested = False
        # Register the signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, _sig, _frame):  # pylint: disable=unused-argument
        """Handle SIGINT (Ctrl+C) signals.
        
        The first Ctrl+C sets the cancellation flag.
        The second Ctrl+C forces an immediate exit.
        
        Args:
            _sig: Signal number (unused)
            _frame: Current stack frame (unused)
        """
        if not self.cancel_requested:
            console.print("\n[bold yellow]Cancellation requested. Cleaning up...[/]")
            self.cancel_requested = True
            # We don't exit immediately - we set a flag and let the program clean up gracefully
        else:
            # If user presses Ctrl+C a second time, exit immediately
            console.print("\n[bold red]Forced exit.[/]")
            sys.exit(1)
    
    def is_cancellation_requested(self):
        """Check if cancellation has been requested.
        
        Returns:
            bool: True if cancellation has been requested, False otherwise.
        """
        return self.cancel_requested
    
    def reset(self):
        """Reset the cancellation flag.
        
        Call this method at the beginning of each streaming operation.
        """
        self.cancel_requested = False
        
# Create a global cancellation handler instance
cancellation = CancellationHandler()


class StreamingTokenDisplay:
    """Display for streaming tokens with real-time counter.
    
    This class provides a live UI that shows:
    1. Token count as they arrive
    2. Estimated tokens per second
    3. Progress indicator
    4. Elapsed time and estimated time remaining
    """
    
    def __init__(self, operation_name: str, model: str):
        """
        Initializes a live token display for a streaming operation.
        
        Args:
            operation_name: The name of the current operation (e.g., "Translation", "Editing").
            model: The model used for the operation.
        """
        self.operation_name = operation_name
        self.model = model
        self.tokens = 0
        self.start_time = None
        self.last_update_time = None
        self.tokens_per_second = 0
        self.live = None
        
    def start(self):
        """
        Starts the live token count display and initializes timing for streaming operations.
        """
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.tokens = 0
        
        # Create a live display that will be updated as tokens arrive
        # Use auto_refresh=False to prevent screen artifacting
        self.live = Live(self._generate_display(), refresh_per_second=4, auto_refresh=False)
        self.live.start()
        
    def update(self, new_tokens: int = 1):
        """
        Increments the token count and updates the live streaming token display.
        
        Args:
            new_tokens: The number of tokens to add to the current count.
        """
        if self.live is None:
            return
            
        self.tokens += new_tokens
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Update tokens per second if we have elapsed at least 0.5 seconds
        if elapsed >= 0.5:
            self.tokens_per_second = self.tokens / (current_time - self.start_time)
            self.last_update_time = current_time
            
        # Update the live display
        # Generate display once and use it to prevent flickering
        display = self._generate_display()
        self.live.update(display)
        
    def stop(self):
        """
        Stops the live token display and cleans up the display instance.
        """
        if self.live is not None:
            self.live.stop()
            self.live = None
            
    def get_elapsed_time(self):
        """
        Returns the elapsed time since the display started as a formatted string.
        
        If the timer has not started, returns "0s".
        """
        if self.start_time is None:
            return "0s"
        
        elapsed = time.time() - self.start_time
        return self._format_time(elapsed)
            
    def _format_time(self, seconds):
        """
        Converts a duration in seconds to a human-readable time string.
        
        Returns:
            A formatted string in 'S.s', 'M:SS', or 'H:MM:SS' format depending on duration.
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes}:{seconds:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}"
    
    def _generate_display(self):
        """
        Generates a Rich Panel displaying live token statistics for the current operation.
        
        Returns:
            A Panel object showing the operation name, model, elapsed time, token count,
            tokens per second, and estimated remaining time if applicable.
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        formatted_time = self._format_time(elapsed)
        
        # Create a text with token information
        text = Text()
        text.append(f"{self.operation_name} with model: ", style="bright_white")
        text.append(f"{self.model}\n", style="cyan")
        
        # Add elapsed time with a clock emoji
        text.append("⏱️ Time: ", style="bright_white")
        text.append(f"{formatted_time}", style="magenta bold")
        
        # Add token count with a counter emoji
        text.append("\n🔢 Tokens: ", style="bright_white")
        text.append(f"{self.tokens:,}", style="green bold")
        
        if elapsed > 1.0:
            # Add speed with a lightning emoji
            text.append("\n⚡ Speed: ", style="bright_white")
            text.append(f"{self.tokens_per_second:.1f} tokens/sec", style="yellow")
            
            # Add estimated time if tokens are flowing
            if self.tokens > 0 and self.tokens_per_second > 0:
                # Calculate tokens per character based on a rough estimate
                # This is very approximate but gives users a rough idea
                estimated_total_tokens = self.tokens * 1.5  # Rough estimate
                estimated_remaining = max(0, estimated_total_tokens - self.tokens)
                
                if estimated_remaining > 0 and self.tokens_per_second > 0:
                    remaining_seconds = estimated_remaining / self.tokens_per_second
                    remaining_formatted = self._format_time(remaining_seconds)
                    text.append("\n⏳ Est. remaining: ", style="bright_white")
                    text.append(f"{remaining_formatted}", style="cyan")
        
        # Create a panel with the text
        panel = Panel(
            text,
            title=f"[bold cyan]{self.operation_name} Progress[/]",
            border_style="blue",
            padding=(1, 2)
        )
        
        return panel


class TranslatorCLI:
    """Command-line interface for the translator."""

    @classmethod
    def setup_openai_client(cls) -> openai.OpenAI:
        """Set up and return an OpenAI client.
        
        Looks for the OpenAI API key in the following locations (in order of precedence):
        1. Environment variables
        2. .env file in the current working directory
        3. .env file in ~/.translator/ directory
        4. .env file in ~/.config/translator/ directory
        """
        # First, try loading from the current directory
        load_dotenv()
        
        # If no API key yet, try user config directories
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try ~/.translator/.env
            home_dir = os.path.expanduser("~")
            translator_dir = os.path.join(home_dir, ".translator")
            env_path = os.path.join(translator_dir, ".env")
            
            if os.path.exists(env_path):
                load_dotenv(env_path)
                api_key = os.getenv("OPENAI_API_KEY")
            
            # If still no API key, try ~/.config/translator/.env
            if not api_key:
                config_dir = os.path.join(home_dir, ".config", "translator")
                env_path = os.path.join(config_dir, ".env")
                
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    api_key = os.getenv("OPENAI_API_KEY")

        # If still no API key, provide helpful error message
        if not api_key:
            console.print(
                "[bold red]Error:[/] OPENAI_API_KEY not found in environment variables or .env files."
            )
            console.print(
                "Please set your OpenAI API key in one of the following locations:"
            )
            console.print("1. Environment variable: export OPENAI_API_KEY=your_api_key")
            console.print("2. Current directory .env file")
            console.print("3. ~/.translator/.env file")
            console.print("4. ~/.config/translator/.env file")
            
            # Offer to create the config directory and a template .env file
            home_dir = os.path.expanduser("~")
            translator_dir = os.path.join(home_dir, ".translator")
            
            if not os.path.exists(translator_dir):
                if cls.confirm("Would you like to create a config directory at ~/.translator?"):
                    try:
                        os.makedirs(translator_dir, exist_ok=True)
                        env_path = os.path.join(translator_dir, ".env")
                        with open(env_path, "w") as f:
                            f.write("# OpenAI API Key (required)\n")
                            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
                            f.write("# Anthropic API Key (optional, for Claude models)\n")
                            f.write("# ANTHROPIC_API_KEY=your_anthropic_api_key_here\n\n")
                            f.write("# Default model (optional, defaults to claude-opus-4.1)\n")
                            f.write("# DEFAULT_MODEL=claude-opus-4.1\n\n")
                            f.write("# Output directory (defaults to same location as input file)\n")
                            f.write("# OUTPUT_DIR=/path/to/output/directory\n\n")
                            f.write("# Log level (defaults to INFO)\n")
                            f.write("# LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL\n")
                        
                        console.print(f"[bold green]Created config template at:[/] {env_path}")
                        console.print("Please edit this file to add your API keys.")
                    except Exception as e:
                        console.print(f"[bold red]Failed to create config directory:[/] {str(e)}")
                
            sys.exit(1)

        return openai.OpenAI(api_key=api_key)

    @classmethod
    def setup_anthropic_client(cls) -> Optional[anthropic.Anthropic]:
        """Set up and return an Anthropic client.

        Looks for the Anthropic API key in the following locations (in order of precedence):
        1. Environment variables
        2. .env file in the current working directory
        3. .env file in ~/.translator/ directory
        4. .env file in ~/.config/translator/ directory

        Returns None if no API key is found (Anthropic models will be unavailable).
        """
        # API key should already be loaded by setup_openai_client
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            # Anthropic is optional, so just return None if no key is found
            return None

        return anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def confirm(prompt: str) -> bool:
        """Ask for user confirmation."""
        response = input(f"{prompt} (y/n): ").strip().lower()
        return response == "y" or response == "yes"

    @staticmethod
    def display_model_info() -> None:
        """Display information about available models and their costs."""
        # Display configured models table
        table = Table(title="Available Models (with pricing)")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="blue")
        table.add_column("Max Tokens", style="green")
        table.add_column("Input Cost (per 1K tokens)", style="yellow")
        table.add_column("Output Cost (per 1K tokens)", style="yellow")
        table.add_column("Capabilities", style="magenta")

        for model, config in ModelConfig.MODELS.items():
            capabilities = config.get('capabilities', [])
            cap_display = ", ".join(capabilities) if capabilities else "Standard"
            provider = config.get('provider', 'unknown').title()
            table.add_row(
                model,
                provider,
                f"{config['max_tokens']:,}",
                f"${config['input_cost']:.4f}",
                f"${config['output_cost']:.4f}",
                cap_display
            )

        console.print(table)

        # Try to fetch and display additional models from OpenAI API
        console.print("\n[bold]Fetching additional models from OpenAI API...[/]")
        try:
            api_models = ModelConfig.get_available_openai_models()
            local_models = set(ModelConfig.MODELS.keys())
            additional_models = [m for m in api_models if m not in local_models]

            if additional_models:
                api_table = Table(title="Additional API Models (no local pricing)")
                api_table.add_column("Model", style="cyan")
                api_table.add_column("Status", style="green")

                for model in additional_models:
                    api_table.add_row(model, "Available via API")

                console.print(api_table)
                console.print(f"\n[yellow]Note:[/] {len(additional_models)} additional models available. Use them with caution as pricing is not configured locally.")
            else:
                console.print("[green]✓[/] All available API models are already configured locally.")

        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not fetch API models: {str(e)}")
            console.print("[dim]Showing locally configured models only.[/]")

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

    @staticmethod
    def get_config_paths() -> list:
        """Get a list of possible configuration file paths.
        
        Returns:
            A list of possible configuration file paths in order of precedence.
        """
        paths = []
        
        # Current directory
        paths.append(os.path.join(os.getcwd(), ".env"))
        
        # User config directories
        home_dir = os.path.expanduser("~")
        paths.append(os.path.join(home_dir, ".translator", ".env"))
        paths.append(os.path.join(home_dir, ".config", "translator", ".env"))
        
        return paths
    
    @classmethod
    def create_config_dialog(cls) -> None:
        """Interactive dialog to create and configure the .env file."""
        # We use the global console object for consistent UI throughout the application
        console.print("[bold]Translator Configuration Setup[/]")
        console.print("This will help you set up your translator configuration.")
        
        # Get list of possible config file paths
        config_paths = cls.get_config_paths()
        
        # Present options to user
        console.print("\n[bold]Select configuration location:[/]")
        for i, path in enumerate(config_paths):
            exists = os.path.exists(path)
            status = "[green]exists[/]" if exists else "[dim]does not exist[/]"
            console.print(f"{i+1}. {path} {status}")
        
        # Get user choice with input validation
        while True:
            choice = input("\nEnter number of preferred location (or press Enter for default ~/.translator/.env): ")
            if not choice:
                # Default to ~/.translator/.env
                choice = 2
                break
            
            try:
                choice = int(choice)
                if 1 <= choice <= len(config_paths):
                    break
                else:
                    console.print(f"[red]Please enter a number between 1 and {len(config_paths)}[/]")
            except ValueError:
                console.print("[red]Please enter a valid number[/]")
        
        selected_path = config_paths[choice-1]
        
        # Make sure the directory exists
        config_dir = os.path.dirname(selected_path)
        if not os.path.exists(config_dir):
            console.print(f"Creating directory: {config_dir}")
            try:
                os.makedirs(config_dir, exist_ok=True)
            except Exception as e:
                console.print(f"[bold red]Error creating directory:[/] {str(e)}")
                return
        
        # Collect configuration values
        config = {}
        
        # OpenAI API Key (required)
        console.print("\n[bold]OpenAI API Key (required)[/]")
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            console.print("[bold red]Error:[/] API key is required.")
            return
        config["OPENAI_API_KEY"] = api_key
        
        # Default model (optional)
        console.print("\n[bold]Default Model (optional, default: claude-opus-4.1)[/]")
        console.print("Available models: " + ", ".join(ModelConfig.MODELS.keys()))
        default_model = input("Enter your preferred default model (or press Enter for claude-opus-4.1): ").strip()
        if default_model:
            if default_model in ModelConfig.MODELS:
                config["DEFAULT_MODEL"] = default_model
            else:
                console.print(f"[yellow]Warning:[/] Unknown model '{default_model}'. Using o3.")
        
        # Write the configuration file
        try:
            with open(selected_path, "w") as f:
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
                
                # Add comments for documentation
                if "DEFAULT_MODEL" not in config:
                    f.write("\n# Default model (uncomment to change)\n# DEFAULT_MODEL=claude-opus-4.1\n")
                
                # Include comment about output directory but don't prompt for it
                f.write("\n# Output directory (defaults to same location as input file)\n# OUTPUT_DIR=/path/to/output/directory\n")
                
                # Include comment about log level but don't prompt for it
                f.write("\n# Log level (defaults to INFO)\n# LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL\n")
            
            console.print(f"[bold green]Configuration saved to:[/] {selected_path}")
            console.print("[bold green]You're all set! You can now use the translator command.[/]")
            
        except Exception as e:
            console.print(f"[bold red]Error saving configuration:[/] {str(e)}")
    
    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Parse command-line arguments.

        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Translate text files to different languages using OpenAI."
        )
        
        # Check first argument to see if it's a known command
        import sys
        first_arg = sys.argv[1] if len(sys.argv) > 1 else None
        
        # If the first argument is 'config', use command subparsers
        if first_arg == 'config':
            subparsers = parser.add_subparsers(dest="command")
            subparsers.add_parser("config", help="Configure the translator")
            
            # Parse args and return
            return parser.parse_args()
            
        # If first argument is 'translate', remove it to make it transparent
        if first_arg == 'translate':
            sys.argv.pop(1)  # Remove 'translate' argument
            
        # Set up standard translation arguments
        parser.add_argument("file", nargs='?', help="Path to the text file to translate")
        parser.add_argument("language", nargs='?', help="Target language for translation")
        parser.add_argument("-o", "--output", help="Output file path (optional)")
        parser.add_argument(
            "-m", "--model", default="claude-opus-4.1", help="AI model to use (default: claude-opus-4.1)"
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
        parser.add_argument(
            "--headless",
            action="store_true",
            help="Skip context gathering and run without user interaction",
        )

        args = parser.parse_args()
        
        # Add command = None to indicate default translation
        args.command = None
        return args

    @classmethod
    def _parse_and_validate_args(
        cls, args: argparse.Namespace
    ) -> Tuple[str, str, Optional[str], str, bool, bool, int, bool, bool, bool]:
        """Parse and validate command line arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            Tuple containing: input_file, target_language, output_file, model,
            skip_edit, do_critique, critique_loops, estimate_only, has_valid_input, headless
        """
        # If --list-models is specified, display model info and exit
        if args.list_models:
            cls.display_model_info()
            sys.exit(0)

        # Check that required arguments are provided for translation
        if not args.file or not args.language:
            console.print("[bold red]Error:[/] file and language arguments are required for translation")
            sys.exit(1)

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
        headless = args.headless

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
            headless,
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
        headless: bool,
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
            headless: Whether running in headless mode

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
            if not headless:
                console.print(
                    f"[bold red]Warning:[/] This text may exceed the {model} model's token limit."
                )
                console.print(
                    "Consider splitting the file into smaller parts or using a model with a higher token limit."
                )
            # Skip confirmation dialog when in estimate-only mode or headless mode
            if not estimate_only and not headless and not cls.confirm("Continue anyway?"):
                should_continue = False
            elif headless and not estimate_only:
                console.print("[dim]Running in headless mode - proceeding despite token limit warning.[/dim]")

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
        """
        Translates frontmatter fields if present and returns the translated data and token usage.
        
        If frontmatter exists and contains translatable fields, translates those fields using the provided Translator instance, optionally displaying a live token count and supporting cancellation. Returns the translated frontmatter, token usage for the frontmatter translation, and cumulative usage.
        
        Args:
            has_frontmatter: Indicates if the input contains frontmatter.
            frontmatter_data: The frontmatter data to translate, if present.
            translator: The Translator instance used for translation.
            target_language: The language to translate the frontmatter into.
            model: The model to use for translation.
        
        Returns:
            A tuple containing the translated frontmatter (or None if not present), a dictionary of token usage for the frontmatter translation, and a dictionary of total token usage.
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
                
                # Use streaming for improved user experience
                use_streaming = True
                
                if use_streaming:
                    # Create token display
                    token_display = StreamingTokenDisplay("Frontmatter Translation", model)
                    
                    # Define a callback to update the token counter
                    def token_callback(_token):  # pylint: disable=unused-argument
                        """
                        Updates the streaming token display during token generation.
                        
                        This callback is intended to be passed to streaming translation or completion methods to refresh the live token count display each time a new token is received.
                        """
                        token_display.update()
                    
                    # Start the token display
                    token_display.start()
                    
                    # Pass cancellation handler and reset it before starting
                    cancellation.reset()
                    translated_frontmatter, frontmatter_usage, error_msg = (
                        translator.translate_frontmatter(
                            frontmatter_data, 
                            translatable_fields, 
                            target_language, 
                            model, 
                            stream=True,
                            cancellation_handler=cancellation,
                            token_callback=token_callback
                        )
                    )
                    
                    # Stop the token display
                    token_display.stop()
                    
                    # Show final count and elapsed time
                    elapsed_time = token_display.get_elapsed_time()
                    console.print(f"[bold green]Frontmatter translation complete![/] Generated [bold]{frontmatter_usage['completion_tokens']:,}[/] tokens in [bold magenta]{elapsed_time}[/]")
                else:
                    # Translate frontmatter fields without streaming
                    translated_frontmatter, frontmatter_usage, error_msg = (
                        translator.translate_frontmatter(
                            frontmatter_data, translatable_fields, target_language, model, stream=False
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
        """
        Translates the main content using the specified model and updates token usage tracking.
        
        Args:
            content_for_translation: The text content to be translated.
            target_language: The language to translate the content into.
            model: The OpenAI model to use for translation.
            total_usage: Dictionary tracking cumulative token usage, updated in place.
        
        Returns:
            A tuple containing the translated content and a dictionary of token usage for this translation step.
        
        Exits the program if a translation error occurs.
        """
        # Use streaming for improved user experience
        use_streaming = True
        
        if use_streaming:
            # Custom progress display with token counter
            console.print("[bold green]Translating...[/]")
            
            # Create token display
            token_display = StreamingTokenDisplay("Translation", model)
            
            # Define a callback to update the token counter
            def token_callback(_token):  # pylint: disable=unused-argument
                """
                Updates the streaming token display during token generation.
                
                This callback is intended to be passed to streaming translation operations to refresh
                the live token count display each time a new token is received.
                """
                token_display.update()
            
            # Start the token display
            token_display.start()
            
            # Pass cancellation handler and reset it before starting
            cancellation.reset()
            translated_content, translation_usage, error_msg = translator.translate_text(
                content_for_translation, 
                target_language, 
                model, 
                stream=True, 
                cancellation_handler=cancellation,
                token_callback=token_callback
            )
            
            # Stop the token display
            token_display.stop()
            
            # Show final count and elapsed time
            elapsed_time = token_display.get_elapsed_time()
            console.print(f"[bold green]Translation complete![/] Received [bold]{translation_usage['completion_tokens']:,}[/] tokens in [bold magenta]{elapsed_time}[/]")
        else:
            # Traditional progress spinner (non-streaming)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Translating...[/]"),
                transient=True,
            ) as progress:
                progress.add_task("translating", total=None)
                
                # Perform translation of main content and get token usage
                translated_content, translation_usage, error_msg = translator.translate_text(
                    content_for_translation, target_language, model, stream=False
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
        """
        Edits the translated content for fluency and accuracy unless editing is skipped.
        
        If editing is performed, displays a live token count during the process and updates token usage statistics. Handles user cancellation and displays warnings if errors occur.
        
        Args:
            skip_edit: If True, skips the editing step.
            translated_content: The content to be edited.
            content_for_translation: The original untranslated content.
            target_language: The language into which the content is being translated.
            model: The model used for editing.
            total_usage: Dictionary tracking cumulative token usage, updated in place.
        
        Returns:
            A tuple containing the (possibly edited) content and a dictionary of token usage for the editing step.
        """
        # Initialize edit usage tracking
        edit_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Perform editing if not skipped
        if not skip_edit:
            console.print("[bold]Editing translation for fluency and accuracy...[/]")
            
            # Use streaming for improved user experience
            use_streaming = True
            
            if use_streaming:
                # Create token display
                token_display = StreamingTokenDisplay("Editing", model)
                
                # Define a callback to update the token counter
                def token_callback(_token):  # pylint: disable=unused-argument
                    """
                    Updates the streaming token display during token generation.
                    
                    Intended as a callback for streaming token events to refresh the live token count UI.
                    """
                    token_display.update()
                
                # Start the token display
                token_display.start()
                
                # Pass cancellation handler and reset it before starting
                cancellation.reset()
                translated_content, edit_usage, error_msg = translator.edit_translation(
                    translated_content, 
                    content_for_translation, 
                    target_language, 
                    model, 
                    stream=True,
                    cancellation_handler=cancellation,
                    token_callback=token_callback
                )
                
                # Stop the token display
                token_display.stop()
                
                # Show final count and elapsed time
                elapsed_time = token_display.get_elapsed_time()
                console.print(f"[bold green]Editing complete![/] Processed [bold]{edit_usage['completion_tokens']:,}[/] tokens in [bold magenta]{elapsed_time}[/]")
            else:
                # Use Rich Progress bar for editing (non-streaming)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold green]Editing translation...[/]"),
                    transient=True,
                ) as progress:
                    progress.add_task("editing", total=None)
                    
                    # Perform editing without streaming
                    translated_content, edit_usage, error_msg = translator.edit_translation(
                        translated_content, content_for_translation, target_language, model, stream=False
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
        """
        Performs one or more critique and revision loops on translated content.
        
        If enabled, iteratively generates critiques of the translated content and applies feedback to improve it, updating token usage statistics for each step. Stores all critiques in the translator's log for later reference.
        
        Args:
            do_critique: Whether to perform critique and revision loops.
            critique_loops: Number of critique/revision cycles to perform.
            translated_content: The content to be critiqued and improved.
            content_for_translation: The original source content.
            target_language: The language into which the content is being translated.
            model: The model used for critique and feedback.
            total_usage: Dictionary tracking cumulative token usage.
        
        Returns:
            A tuple containing:
                - The improved content after all critique loops.
                - Token usage for the last critique step.
                - Token usage for the last feedback application.
                - List of token usage dictionaries for each critique step.
                - List of token usage dictionaries for each feedback application.
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
                
                # Use streaming for improved user experience
                use_streaming = True
                
                if use_streaming:
                    # Create token display
                    token_display = StreamingTokenDisplay("Critique Generation", model)
                    
                    # Define a callback to update the token counter
                    def token_callback(_token):  # pylint: disable=unused-argument
                        """
                        Updates the streaming token display during token generation.
                        
                        This callback is intended to be passed to streaming translation operations to refresh
                        the live token count display each time a new token is received.
                        """
                        token_display.update()
                    
                    # Start the token display
                    token_display.start()
                    
                    # Pass cancellation handler and reset it before starting
                    cancellation.reset()
                    _, loop_critique_usage, critique_feedback, error_msg = (
                        translator.critique_translation(
                            translated_content,
                            content_for_translation,
                            target_language,
                            model,
                            stream=True,
                            cancellation_handler=cancellation,
                            token_callback=token_callback
                        )
                    )
                    
                    # Stop the token display
                    token_display.stop()
                    
                    # Show final count and elapsed time
                    elapsed_time = token_display.get_elapsed_time()
                    console.print(f"[bold green]Critique complete![/] Generated [bold]{loop_critique_usage['completion_tokens']:,}[/] tokens in [bold magenta]{elapsed_time}[/]")
                else:
                    # Use Rich Progress bar for critique generation (non-streaming)
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold red]Generating critique...[/]"),
                        transient=True,
                    ) as progress:
                        progress.add_task("critiquing", total=None)
                        
                        # Generate critique without streaming
                        _, loop_critique_usage, critique_feedback, error_msg = (
                            translator.critique_translation(
                                translated_content,
                                content_for_translation,
                                target_language,
                                model,
                                stream=False
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
                
                # Use streaming for improved user experience
                use_streaming = True
                
                if use_streaming:
                    # Create token display
                    token_display = StreamingTokenDisplay("Critique Application", model)
                    
                    # Define a callback to update the token counter
                    def token_callback(_token):  # pylint: disable=unused-argument
                        """
                        Updates the streaming token display when a new token is received during streaming operations.
                        """
                        token_display.update()
                    
                    # Start the token display
                    token_display.start()
                    
                    # Pass cancellation handler and reset it before starting
                    cancellation.reset()
                    translated_content, loop_feedback_usage, error_msg = (
                        translator.apply_critique_feedback(
                            translated_content,
                            content_for_translation,
                            critique_feedback,
                            target_language,
                            model,
                            stream=True,
                            cancellation_handler=cancellation,
                            token_callback=token_callback
                        )
                    )
                    
                    # Stop the token display
                    token_display.stop()
                    
                    # Show final count and elapsed time
                    elapsed_time = token_display.get_elapsed_time()
                    console.print(f"[bold green]Critique application complete![/] Generated [bold]{loop_feedback_usage['completion_tokens']:,}[/] tokens in [bold magenta]{elapsed_time}[/]")
                else:
                    # Use Rich Progress bar for applying feedback (non-streaming)
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]Applying critique feedback...[/]"),
                        transient=True,
                    ) as progress:
                        progress.add_task("applying_feedback", total=None)
                        
                        # Apply feedback without streaming
                        translated_content, loop_feedback_usage, error_msg = (
                            translator.apply_critique_feedback(
                                translated_content,
                                content_for_translation,
                                critique_feedback,
                                target_language,
                                model,
                                stream=False
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
            "translation_context": translator.translation_context,
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

        # Generate the narrative (using non-streaming for this since it's less critical)
        narrative = log_interpreter.generate_narrative(log_data, "o4-mini", stream=False)

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
        headless: bool,
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
            headless: Whether running in headless mode

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
        
        # Handle context gathering based on headless flag
        if headless:
            # Skip context gathering in headless mode
            console.print("[dim]Running in headless mode - skipping context gathering.[/dim]")
            translator.translation_context = ""
        else:
            # Ask user for context about the piece being translated
            console.print("[bold cyan]Please provide some context about the piece being translated.[/]")
            console.print("This could include information about the author, intended audience, tone, purpose, etc.")
            console.print("This context will help produce a more accurate and appropriate translation.")
            console.print("(Press Enter twice to submit)")
            
            # Collect context input (allowing for multi-line input)
            context_lines = []
            while True:
                line = input()
                if not line and (not context_lines or not context_lines[-1]):
                    break
                context_lines.append(line)
            
            translation_context = "\n".join(context_lines).strip()
            
            # If context is provided, show confirmation
            if translation_context:
                console.print("[bold green]Context received. This will be used to guide the translation.[/]")
                # Store context in translator instance for use in prompts
                translator.translation_context = translation_context
            else:
                console.print("[dim]No context provided. Proceeding with translation.[/dim]")
                translator.translation_context = ""

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

        # Handle "config" command
        if hasattr(args, 'command') and args.command == 'config':
            cls.create_config_dialog()
            return
        
        # Handle "list-models" command
        if hasattr(args, 'list_models') and args.list_models:
            cls.display_model_info()
            sys.exit(0)

        # Handle translation (default action)
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
            headless,
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
                headless,
            )
        )

        if not should_continue:
            sys.exit(0)

        # If only estimating, exit now
        if estimate_only:
            sys.exit(0)

        # Set up OpenAI and Anthropic clients
        openai_client = cls.setup_openai_client()
        anthropic_client = cls.setup_anthropic_client()
        translator = Translator(openai_client=openai_client, anthropic_client=anthropic_client)

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
            headless,
        )
