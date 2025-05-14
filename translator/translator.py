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

console = Console()


class Translator:
    """Core translation logic using OpenAI's API."""

    def __init__(self, client: openai.OpenAI):
        """Initialize the translator.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client

    def translate_text(self, text: str, target_language: str, model: str) -> Tuple[str, Dict]:
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
        system_prompt = f"""
        1. Read the provided text carefully, preserving all formatting, markdown, and structure exactly as they appear.
        2. Identify any block quotes and code blocks.
        3. Do not translate text in block quotes or in code blocks (including text within code blocks).
        4. Translate everything else into {target_language}.
        5. Maintain the original formatting, markdown, and structure in your output.
        6. Provide a natural-sounding translation rather than a word-for-word one.
        7. For idioms, colloquialisms, or slang, render them in an equivalent, natural way in {target_language} whenever possible.
        8. If there isn't a direct or natural translation for a particular term or phrase, keep it in the original language and surround it with quotes if necessary.
        9. Ensure that technical terms or jargon remain accurate; if there's no suitable translation, keep the original term.
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
                
                response = self.client.chat.completions.create(
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

    def edit_translation(self, translated_text: str, original_text: str, target_language: str, model: str) -> Tuple[str, Dict]:
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
        system_prompt = f"""
        1. Carefully read the translated text alongside the original text in its entirety.
        2. Compare both texts to ensure the translation accurately reflects the original meaning.
        3. Correct any grammatical errors you find in the {target_language} text.
        4. Adjust phrasing to make it sound natural and fluent for {target_language} speakers, making sure idioms and expressions are culturally appropriate.
        5. Preserve the original tone, nuance, and style, including any formatting, markdown, and structure.
        6. Avoid adding new information or altering the core meaning.
        7. Ensure the final result doesn't feel machine-translated but remains faithful to the source.
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
                
                response = self.client.chat.completions.create(
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

    def critique_translation(self, translated_text: str, original_text: str, target_language: str, model: str) -> Tuple[str, Dict, str]:
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
        system_prompt = f"""You are a highly critical professional translator and linguistic expert specializing in {target_language}.
Your task is to ruthlessly critique the translation by:

1. Meticulously comparing the translated text with the original, identifying ANY inaccuracies, mistranslations, or omissions
2. Highlighting nuances, cultural references, or idioms that were lost or mistranslated
3. Scrutinizing for grammatical errors, awkward phrasing, or unnatural expressions in {target_language}
4. Checking for inconsistencies in tone, style, or register compared to the original
5. Verifying that technical terms are translated accurately and consistently
6. Ensuring no content was accidentally skipped or added
7. Finding places where the translation sounds machine-like or overly literal

Be extremely thorough and critical in your assessment. Do not accept mediocre translations.
List specific issues and suggestions for improvement, organized by severity and category.
Your critique should be detailed enough for another translator to address all the issues.

Your goal is to help create a perfect translation that reads as if originally written in {target_language} while being 100% faithful to the source.
"""
        
        user_prompt = f"""# ORIGINAL TEXT
{original_text}

# CURRENT TRANSLATION
{translated_text}

Please critique this translation mercilessly and provide detailed feedback on what needs to be improved.
Format your critique as a structured list of issues, organized by severity and category.
Include specific suggestions for how to fix each issue."""
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold red]Generating critique...[/]"),
                transient=True
            ) as progress:
                progress.add_task("critiquing", total=None)
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                
                # Extract usage statistics
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                critique_feedback = response.choices[0].message.content
                
                return translated_text, usage, critique_feedback
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Critique failed: {escape(str(e))}")
            # Return original translation if critique fails with empty usage stats
            return translated_text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, ""
            
    def apply_critique_feedback(self, translated_text: str, original_text: str, critique_feedback: str, target_language: str, model: str) -> Tuple[str, Dict]:
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
        system_prompt = f"""You are a master translator and editor specializing in {target_language}.
Your task is to improve a translation based on detailed critique feedback.

1. Carefully read the original text, current translation, and the critique feedback
2. Address ALL issues identified in the critique
3. Apply the specific suggestions for improvement
4. Ensure the translation is accurate, natural-sounding, and faithful to the original
5. Preserve all formatting, markdown, and structure of the original text
6. Make sure the final text reads as if it were originally written in {target_language}

Do not ignore any of the critique points. Every issue identified must be addressed in your improved version.
"""
        
        user_prompt = f"""# ORIGINAL TEXT
{original_text}

# CURRENT TRANSLATION
{translated_text}

# CRITIQUE FEEDBACK
{critique_feedback}

Please address ALL issues identified in the critique and provide an improved translation.
Return ONLY the improved translated text without explanations or comments."""
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Applying critique feedback...[/]"),
                transient=True
            ) as progress:
                progress.add_task("applying_feedback", total=None)
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.5
                )
                
                # Extract usage statistics
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                return response.choices[0].message.content, usage
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Failed to apply critique feedback: {escape(str(e))}")
            # Return original translation if applying critique feedback fails
            return translated_text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def translate_frontmatter(self, frontmatter_data: Dict, fields: List[str], target_language: str, model: str) -> Tuple[Dict, Dict]:
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
            return translated_frontmatter, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Prepare text for translation
        fields_text = ""
        for field in fields:
            fields_text += f"{field}: {frontmatter_data[field]}\n\n"
        
        system_prompt = f"""You are a professional translator. Translate the following frontmatter fields to {target_language}.
Each field is in the format "field_name: content". Translate ONLY the content, not the field names.
Return the translated content in the exact same format, preserving all field names."""
        
        try:
            console.print(f"[bold]Translating frontmatter fields:[/] {', '.join(fields)}")
            
            response = self.client.chat.completions.create(
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
            return frontmatter_data, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}