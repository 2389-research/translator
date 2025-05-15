#!/usr/bin/env python3
# ABOUTME: Core translation logic using OpenAI's API.
# ABOUTME: Provides translation, editing, and critique functions.

import sys
import re
from typing import Dict, List, Tuple

import openai
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn

from translator.prompts import Prompts

console = Console()


class Translator:
    """Core translation logic using OpenAI's API."""

    def __init__(self, client: openai.OpenAI):
        """Initialize the translator.

        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.translation_log = {
            "translation": {},
            "editing": {},
            "critique": {},
            "feedback": {},
            "frontmatter": {},
            "all_critiques": [],
        }

    def translate_text(
        self, text: str, target_language: str, model: str
    ) -> Tuple[str, Dict]:
        """Translate text to the target language using OpenAI.

        Args:
            text: The text to translate
            target_language: The target language for translation
            model: The model to use for translation

        Returns:
            Tuple containing:
                - Translated text
                - Usage information dictionary

        Raises:
            SystemExit: If translation fails
        """
        system_prompt = Prompts.translation_system_prompt(target_language)
        user_prompt = Prompts.translation_user_prompt(text)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Translating...[/]"),
                transient=True,
            ) as progress:
                progress.add_task("translating", total=None)

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                # Extract usage statistics
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                # Log the translation prompts and response
                self.translation_log["translation"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response.choices[0].message.content,
                    "usage": usage,
                }

                return response.choices[0].message.content, usage
        except Exception as e:
            console.print(f"[bold red]Error:[/] Translation failed: {escape(str(e))}")
            sys.exit(1)

    def edit_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str
    ) -> Tuple[str, Dict]:
        """Edit the translation to ensure it makes sense in the target language while preserving original meaning.

        Args:
            translated_text: The translated text to edit
            original_text: The original text for reference
            target_language: The target language
            model: The model to use for editing

        Returns:
            Tuple containing:
                - Edited text
                - Usage information dictionary
        """
        system_prompt = Prompts.editing_system_prompt(target_language)
        user_prompt = Prompts.editing_user_prompt(
            original_text, translated_text, target_language
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Editing translation...[/]"),
                transient=True,
            ) as progress:
                progress.add_task("editing", total=None)

                # Create parameters for API call
                params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }

                # Add top_p for models that support it
                if model != "o3":
                    params["top_p"] = 1.0

                response = self.client.chat.completions.create(**params)

                # Extract usage statistics
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                # Log the editing prompts and response
                self.translation_log["editing"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response.choices[0].message.content,
                    "usage": usage,
                }

                return response.choices[0].message.content, usage
        except Exception as e:
            console.print(f"[bold red]Error:[/] Editing failed: {escape(str(e))}")
            # Return original translation if editing fails with empty usage stats
            return translated_text, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def critique_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str
    ) -> Tuple[str, Dict, str]:
        """Aggressively critique the translation against the original text.

        Args:
            translated_text: The translated text to critique
            original_text: The original text for reference
            target_language: The target language
            model: The model to use for critique

        Returns:
            Tuple containing:
                - Translated text (unchanged)
                - Usage information dictionary
                - Critique feedback as a string
        """
        system_prompt = Prompts.critique_system_prompt(target_language)
        user_prompt = Prompts.critique_user_prompt(original_text, translated_text)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold red]Generating critique...[/]"),
                transient=True,
            ) as progress:
                progress.add_task("critiquing", total=None)

                # Create parameters without temperature for o3 model
                params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }

                # Add temperature for models other than o3
                if model != "o3":
                    params["temperature"] = 0.7

                response = self.client.chat.completions.create(**params)

                # Extract usage statistics
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                critique_feedback = response.choices[0].message.content

                # Log the critique prompts and response
                self.translation_log["critique"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": critique_feedback,
                    "usage": usage,
                }

                return translated_text, usage, critique_feedback
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Critique failed: {escape(str(e))}")
            # Return original translation if critique fails with empty usage stats
            return (
                translated_text,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "",
            )

    def apply_critique_feedback(
        self,
        translated_text: str,
        original_text: str,
        critique_feedback: str,
        target_language: str,
        model: str,
    ) -> Tuple[str, Dict]:
        """Apply critique feedback to improve the translation.

        Args:
            translated_text: The translated text to improve
            original_text: The original text for reference
            critique_feedback: The detailed critique feedback
            target_language: The target language
            model: The model to use for applying feedback

        Returns:
            Tuple containing:
                - Improved text after applying critique feedback
                - Usage information dictionary
        """
        system_prompt = Prompts.feedback_system_prompt(target_language)
        user_prompt = Prompts.feedback_user_prompt(
            original_text, translated_text, critique_feedback
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Applying critique feedback...[/]"),
                transient=True,
            ) as progress:
                progress.add_task("applying_feedback", total=None)

                # Create parameters without temperature for o3 model
                params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }

                # Add temperature for models other than o3
                if model != "o3":
                    params["temperature"] = 0.5

                response = self.client.chat.completions.create(**params)

                # Extract usage statistics
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                # Log the feedback application prompts and response
                self.translation_log["feedback"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response.choices[0].message.content,
                    "usage": usage,
                }

                return response.choices[0].message.content, usage
        except Exception as e:
            console.print(
                f"[bold yellow]Warning:[/] Failed to apply critique feedback: {escape(str(e))}"
            )
            # Return original translation if applying critique feedback fails
            return translated_text, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def translate_frontmatter(
        self,
        frontmatter_data: Dict,
        fields: List[str],
        target_language: str,
        model: str,
    ) -> Tuple[Dict, Dict]:
        """Translate specified fields in frontmatter.

        Args:
            frontmatter_data: The frontmatter data as a dictionary
            fields: List of fields to translate
            target_language: The target language
            model: The model to use for translation

        Returns:
            Tuple containing:
                - Translated frontmatter data
                - Usage information dictionary
        """
        # Create a copy to avoid modifying the original
        translated_frontmatter = frontmatter_data.copy()

        if not fields:
            return translated_frontmatter, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        # Prepare text for translation
        fields_text = ""
        for field in fields:
            fields_text += f"{field}: {frontmatter_data[field]}\n\n"

        system_prompt = Prompts.frontmatter_system_prompt(target_language)
        user_prompt = Prompts.frontmatter_user_prompt(fields_text)

        try:
            console.print(
                f"[bold]Translating frontmatter fields:[/] {', '.join(fields)}"
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            # Extract usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Parse the response to get translated fields
            translated_text = response.choices[0].message.content

            # Log the frontmatter translation prompts and response
            self.translation_log["frontmatter"] = {
                "model": model,
                "target_language": target_language,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": translated_text,
                "usage": usage,
                "fields": fields,
            }

            # Extract each translated field from the response
            for field in fields:
                pattern = rf"{field}: (.*?)(?:\n\n|\n$|$)"
                match = re.search(pattern, translated_text, re.DOTALL)
                if match:
                    translated_value = match.group(1).strip()
                    translated_frontmatter[field] = translated_value

            return translated_frontmatter, usage
        except Exception as e:
            console.print(
                f"[bold yellow]Warning:[/] Failed to translate frontmatter: {escape(str(e))}"
            )
            # Return original frontmatter on error
            return frontmatter_data, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
