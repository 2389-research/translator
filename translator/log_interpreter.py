#!/usr/bin/env python3
# ABOUTME: Module for interpreting translation log files.
# ABOUTME: Provides functions to generate narrative summaries of translation logs.

import json
from pathlib import Path
from typing import Dict, Optional

import openai
from rich.console import Console
from rich.markup import escape

console = Console()


class LogInterpreter:
    """Interprets translation log files and generates narrative summaries."""

    def __init__(self, client: openai.OpenAI):
        """Initialize the log interpreter.

        Args:
            client: OpenAI client instance
        """
        self.client = client

    @staticmethod
    def read_log_file(log_path: str) -> Optional[Dict]:
        """Read and parse a JSON log file.

        Args:
            log_path: Path to the log file

        Returns:
            Parsed log data as a dictionary, or None if the file cannot be read or parsed
        """
        try:
            with open(log_path, "r", encoding="utf-8") as file:
                return json.loads(file.read())
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            console.print(
                f"[bold red]Error:[/] Failed to read or parse log file: {escape(str(e))}"
            )
            return None

    def generate_narrative(self, log_data: Dict, model: str = "o4-mini") -> str:
        """Generate a narrative interpretation of the translation log.

        Args:
            log_data: The log data dictionary
            model: The model to use for interpretation (default: o4-mini)

        Returns:
            A narrative interpretation of the translation process
        """
        # Extract key information from the log
        target_language = log_data.get("target_language", "unknown")
        language_code = log_data.get("language_code", "unknown")
        translation_model = log_data.get("model", "unknown")
        has_frontmatter = log_data.get("has_frontmatter", False)
        critique_loops = log_data.get("critique_loops", 0)
        do_critique = log_data.get("do_critique", False)
        token_usage = log_data.get("token_usage", {})

        prompts_and_responses = log_data.get("prompts_and_responses", {})

        # Create a system prompt for the interpretation
        system_prompt = (
            "You are an expert translator analyst. Your task is to analyze a translation process "
            "and provide a concise, insightful narrative about what happened during the translation. "
            "Focus on the quality improvements made during the process, challenges encountered, "
            "and the overall effectiveness of the translation workflow."
        )

        # Create a user prompt with specific details from the log
        user_prompt = f"""
Analyze this translation process from the log data:

- Source language: (inferred from content)
- Target language: {target_language} ({language_code})
- Translation model: {translation_model}
- Included frontmatter: {'Yes' if has_frontmatter else 'No'}
- Used critique process: {'Yes, ' + str(critique_loops) + ' loops' if do_critique else 'No'}
- Total tokens used: {token_usage.get('total_tokens', 0):,}

Translation system prompt: {prompts_and_responses.get('translation', {}).get('system_prompt', 'N/A')}

First 300 characters of translation response: 
{prompts_and_responses.get('translation', {}).get('response', 'N/A')[:300]}...

{'Editing system prompt: ' + prompts_and_responses.get('editing', {}).get('system_prompt', 'N/A') if prompts_and_responses.get('editing') else ''}

{'First 300 characters of edited response: ' + prompts_and_responses.get('editing', {}).get('response', 'N/A')[:300] + '...' if prompts_and_responses.get('editing') else ''}

"""

        # Add critique information if available
        if do_critique and prompts_and_responses.get("all_critiques"):
            critiques = prompts_and_responses.get("all_critiques", [])
            user_prompt += f"\nNumber of critique loops: {len(critiques)}\n\n"

            for i, critique in enumerate(
                critiques[:2]
            ):  # Just include the first two critiques to keep prompt size manageable
                user_prompt += (
                    f"Critique {i+1} summary (first 300 chars): {critique[:300]}...\n\n"
                )

        user_prompt += """
Based on this data, provide a concise narrative interpretation (300-500 words) of what happened during the translation process. Include:

1. A summary of the translation workflow
2. Insights about the quality improvements at each stage
3. Analysis of any challenges or issues identified in the critiques
4. Overall assessment of the translation quality and process effectiveness

Your narrative should be informative and helpful to someone who wants to understand what happened during the translation process without getting too technical.
"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            console.print(
                f"[bold red]Error:[/] Failed to generate narrative: {escape(str(e))}"
            )
            return f"Error generating narrative interpretation: {str(e)}"

    @staticmethod
    def write_narrative(output_path: str, narrative: str) -> None:
        """Write the narrative interpretation to a file.

        Args:
            output_path: Path to write the narrative file
            narrative: The narrative text to write
        """
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(narrative)
        except IOError as e:
            console.print(
                f"[bold red]Error:[/] Failed to write narrative file: {escape(str(e))}"
            )

    @staticmethod
    def get_narrative_filename(log_path: str) -> str:
        """Generate a filename for the narrative file based on the log file path.

        Args:
            log_path: Path to the log file

        Returns:
            Path for the narrative file
        """
        log_file = Path(log_path)

        # Extract the base filename (without any extensions)
        parts = log_file.stem.split(".")
        if len(parts) >= 3:  # filename.languagecode.ext.log.json pattern
            # Get the base filename parts (up to languagecode)
            base_parts = parts[:2]  # This gets [filename, languagecode]
            base = ".".join(base_parts)
            return str(log_file.with_name(f"{base}.log"))
        else:
            # Fallback for simple cases - replace extension with .log
            base = log_file.stem.split(".")[0]  # Get the first part as base
            return str(log_file.with_name(f"{base}.log"))
