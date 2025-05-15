#!/usr/bin/env python3
# ABOUTME: Core translation logic using OpenAI's API.
# ABOUTME: Provides translation, editing, and critique functions.

import re
from typing import Dict, List, Optional, Tuple

import openai

from translator.prompts import Prompts


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
    ) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Translate text to the target language using OpenAI.

        Args:
            text: The text to translate
            target_language: The target language for translation
            model: The model to use for translation

        Returns:
            Tuple containing:
                - Translated text (None if error occurred)
                - Usage information dictionary
                - Error message (None if successful)
        """
        system_prompt = Prompts.translation_system_prompt(target_language)
        user_prompt = Prompts.translation_user_prompt(text)
        
        empty_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
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

            return response.choices[0].message.content, usage, None
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            return None, empty_usage, error_msg

    def edit_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str
    ) -> Tuple[str, Dict, Optional[str]]:
        """Edit the translation to ensure it makes sense in the target language while preserving original meaning.

        Args:
            translated_text: The translated text to edit
            original_text: The original text for reference
            target_language: The target language
            model: The model to use for editing

        Returns:
            Tuple containing:
                - Edited text (original text returned if error occurred)
                - Usage information dictionary
                - Error message (None if successful)
        """
        system_prompt = Prompts.editing_system_prompt(target_language)
        user_prompt = Prompts.editing_user_prompt(
            original_text, translated_text, target_language
        )
        
        empty_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
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

            return response.choices[0].message.content, usage, None
        except Exception as e:
            error_msg = f"Editing failed: {str(e)}"
            # Return original translation if editing fails with empty usage stats
            return translated_text, empty_usage, error_msg

    def critique_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str
    ) -> Tuple[str, Dict, str, Optional[str]]:
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
                - Critique feedback as a string (empty if error occurred)
                - Error message (None if successful)
        """
        system_prompt = Prompts.critique_system_prompt(target_language)
        user_prompt = Prompts.critique_user_prompt(original_text, translated_text)
        
        empty_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
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

            return translated_text, usage, critique_feedback, None
        except Exception as e:
            error_msg = f"Critique failed: {str(e)}"
            # Return original translation if critique fails with empty usage stats
            return translated_text, empty_usage, "", error_msg

    def apply_critique_feedback(
        self,
        translated_text: str,
        original_text: str,
        critique_feedback: str,
        target_language: str,
        model: str,
    ) -> Tuple[str, Dict, Optional[str]]:
        """Apply critique feedback to improve the translation.

        Args:
            translated_text: The translated text to improve
            original_text: The original text for reference
            critique_feedback: The detailed critique feedback
            target_language: The target language
            model: The model to use for applying feedback

        Returns:
            Tuple containing:
                - Improved text after applying critique feedback (original text returned if error occurred)
                - Usage information dictionary
                - Error message (None if successful)
        """
        system_prompt = Prompts.feedback_system_prompt(target_language)
        user_prompt = Prompts.feedback_user_prompt(
            original_text, translated_text, critique_feedback
        )
        
        empty_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
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

            return response.choices[0].message.content, usage, None
        except Exception as e:
            error_msg = f"Failed to apply critique feedback: {str(e)}"
            # Return original translation if applying critique feedback fails
            return translated_text, empty_usage, error_msg

    def translate_frontmatter(
        self,
        frontmatter_data: Dict,
        fields: List[str],
        target_language: str,
        model: str,
    ) -> Tuple[Dict, Dict, Optional[str]]:
        """Translate specified fields in frontmatter.

        Args:
            frontmatter_data: The frontmatter data as a dictionary
            fields: List of fields to translate
            target_language: The target language
            model: The model to use for translation

        Returns:
            Tuple containing:
                - Translated frontmatter data (original data returned if error occurred)
                - Usage information dictionary
                - Error message (None if successful)
        """
        # Create a copy to avoid modifying the original
        translated_frontmatter = frontmatter_data.copy()
        
        empty_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if not fields:
            return translated_frontmatter, empty_usage, None

        # Prepare text for translation
        fields_text = ""
        for field in fields:
            fields_text += f"{field}: {frontmatter_data[field]}\n\n"

        system_prompt = Prompts.frontmatter_system_prompt(target_language)
        user_prompt = Prompts.frontmatter_user_prompt(fields_text)

        try:
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

            return translated_frontmatter, usage, None
        except Exception as e:
            error_msg = f"Failed to translate frontmatter: {str(e)}"
            # Return original frontmatter on error
            return frontmatter_data, empty_usage, error_msg
