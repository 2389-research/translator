#!/usr/bin/env python3
# ABOUTME: Core translation logic using multi-provider AI APIs.
# ABOUTME: Provides translation, editing, and critique functions.

import re
from typing import Dict, List, Optional, Tuple

import openai
import anthropic

from translator.prompts import Prompts
from translator.providers import ProviderFactory


class Translator:
    """Core translation logic using multi-provider AI APIs.

    This class provides methods for translating text, editing translations,
    critiquing translations, and applying critique feedback. All methods support
    both streaming and non-streaming responses from OpenAI and Anthropic APIs.

    Streaming responses provide the following benefits:
    1. Lower latency for displaying initial results
    2. More responsive user experience
    3. Possibility of cancelling long-running requests
    """

    def __init__(self, openai_client: openai.OpenAI = None, anthropic_client: anthropic.Anthropic = None):
        """Initialize the translator.

        Args:
            openai_client: OpenAI client instance
            anthropic_client: Anthropic client instance
        """
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.translation_context = ""
        self.translation_log = {
            "translation": {},
            "editing": {},
            "critique": {},
            "feedback": {},
            "frontmatter": {},
            "all_critiques": [],
        }

    def translate_text(
        self, text: str, target_language: str, model: str, stream: bool = False,
        cancellation_handler=None, token_callback=None
    ) -> Tuple[Optional[str], Dict, Optional[str]]:
        """
        Translates text into the target language using the specified AI model.

        Supports both streaming and non-streaming responses, with optional cancellation and token callbacks. Returns the translated text, usage statistics, and an error message if the translation fails.

        Args:
            text: The text to translate.
            target_language: The language to translate the text into.
            model: The AI model to use for translation.
            stream: If True, streams the translation response incrementally.
            cancellation_handler: Optional handler to interrupt translation if cancellation is requested.
            token_callback: Optional function called with each token during streaming.

        Returns:
            A tuple containing:
                - The translated text, or None if an error occurred.
                - A dictionary with usage statistics.
                - An error message string, or None if successful.
        """
        system_prompt = Prompts.translation_system_prompt(target_language)

        try:
            provider = ProviderFactory.create_provider(
                model,
                openai_client=self.openai_client,
                anthropic_client=self.anthropic_client
            )

            translated_text, usage, error = provider.translate_text(
                text=text,
                target_language=target_language,
                model=model,
                system_prompt=system_prompt,
                stream=stream,
                cancellation_handler=cancellation_handler,
                token_callback=token_callback
            )

            # Log the translation prompts and response
            if translated_text:
                user_prompt = f"Translate this text to {target_language}:\n\n{text}"
                self.translation_log["translation"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "context": self.translation_context,
                    "response": translated_text,
                    "usage": usage,
                    "streaming": stream,
                }

            return translated_text, usage, error

        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            empty_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            return None, empty_usage, error_msg

    def edit_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str,
        stream: bool = False, cancellation_handler=None, token_callback=None
    ) -> Tuple[str, Dict, Optional[str]]:
        """
        Edits a translated text to improve fluency and accuracy while preserving the original meaning.

        If streaming is enabled, the response is returned incrementally and can be canceled or processed token by token via optional handlers.

        Args:
            translated_text: The text to be edited.
            original_text: The original source text for reference.
            target_language: The language into which the text is translated.
            model: The model identifier to use for editing.
            stream: If True, enables streaming of the response (default: False).
            cancellation_handler: Optional handler to interrupt the operation if cancellation is requested.
            token_callback: Optional function called with each token during streaming.

        Returns:
            A tuple containing the edited text (or the original if an error occurs), a dictionary with usage statistics, and an error message (None if successful).
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
            provider = ProviderFactory.create_provider(
                model,
                openai_client=self.openai_client,
                anthropic_client=self.anthropic_client
            )

            # Create custom prompt that combines user and text content
            edit_text = f"Edit this translation to improve fluency and accuracy:\n\nOriginal: {original_text}\n\nTranslation: {translated_text}"

            edited_text, usage, error = provider.translate_text(
                text=edit_text,
                target_language=target_language,
                model=model,
                system_prompt=system_prompt,
                stream=stream,
                cancellation_handler=cancellation_handler,
                token_callback=token_callback
            )

            # Log the editing prompts and response
            if edited_text:
                self.translation_log["editing"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": edited_text,
                    "usage": usage,
                    "streaming": stream,
                }
                return edited_text, usage, None
            else:
                return translated_text, empty_usage, error

        except Exception as e:
            error_msg = f"Editing failed: {str(e)}"
            # Return original translation if editing fails with empty usage stats
            return translated_text, empty_usage, error_msg

    def critique_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str,
        stream: bool = False, cancellation_handler=None, token_callback=None
    ) -> Tuple[str, Dict, str, Optional[str]]:
        """
        Provides an aggressive critique of a translated text compared to the original.

        Evaluates the quality and accuracy of the translated text by generating detailed feedback using the specified model. Supports both streaming and non-streaming responses, with optional cancellation and token callbacks.

        Args:
            translated_text: The translated text to be critiqued.
            original_text: The original source text for comparison.
            target_language: The language into which the text was translated.
            model: The model used for critique.
            stream: If True, streams the critique response incrementally.
            cancellation_handler: Optional handler to interrupt streaming if cancellation is requested.
            token_callback: Optional function called with each token during streaming.

        Returns:
            A tuple containing:
                - The original translated text (unchanged).
                - A dictionary with token usage statistics.
                - The critique feedback as a string (empty if an error occurred).
                - An error message if the critique failed, otherwise None.
        """
        system_prompt = Prompts.critique_system_prompt(target_language)
        user_prompt = Prompts.critique_user_prompt(original_text, translated_text)

        empty_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            provider = ProviderFactory.create_provider(
                model,
                openai_client=self.openai_client,
                anthropic_client=self.anthropic_client
            )

            # Create critique text that includes both original and translation
            critique_text = f"Critique this translation:\n\nOriginal: {original_text}\n\nTranslation: {translated_text}"

            critique_feedback, usage, error = provider.translate_text(
                text=critique_text,
                target_language=target_language,
                model=model,
                system_prompt=system_prompt,
                stream=stream,
                cancellation_handler=cancellation_handler,
                token_callback=token_callback
            )

            # Log the critique prompts and response
            if critique_feedback:
                self.translation_log["critique"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": critique_feedback,
                    "usage": usage,
                    "streaming": stream,
                }
                return translated_text, usage, critique_feedback, None
            else:
                return translated_text, empty_usage, "", error

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
        stream: bool = False,
        cancellation_handler=None,
        token_callback=None,
    ) -> Tuple[str, Dict, Optional[str]]:
        """
        Applies critique feedback to improve a translated text using the specified model.

        If streaming is enabled, the response is returned incrementally and can be canceled or processed token-by-token via optional handlers.

        Args:
            translated_text: The translated text to be improved.
            original_text: The original source text for reference.
            critique_feedback: Feedback detailing issues or suggestions for improvement.
            target_language: The language into which the text is being translated.
            model: The model identifier to use for applying feedback.
            stream: If True, enables streaming of the response (default: False).
            cancellation_handler: Optional handler to interrupt processing if cancellation is requested.
            token_callback: Optional function called with each token during streaming.

        Returns:
            A tuple containing:
                - The improved translation (or the original text if an error occurs).
                - A dictionary with token usage statistics.
                - An error message if an error occurred, otherwise None.
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
            provider = ProviderFactory.create_provider(
                model,
                openai_client=self.openai_client,
                anthropic_client=self.anthropic_client
            )

            # Create feedback text that includes original, translation, and critique
            feedback_text = f"Apply this feedback to improve the translation:\n\nOriginal: {original_text}\n\nTranslation: {translated_text}\n\nFeedback: {critique_feedback}"

            improved_text, usage, error = provider.translate_text(
                text=feedback_text,
                target_language=target_language,
                model=model,
                system_prompt=system_prompt,
                stream=stream,
                cancellation_handler=cancellation_handler,
                token_callback=token_callback
            )

            # Log the feedback application prompts and response
            if improved_text:
                self.translation_log["feedback"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": improved_text,
                    "usage": usage,
                    "streaming": stream,
                }
                return improved_text, usage, None
            else:
                return translated_text, empty_usage, error

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
        stream: bool = False,
        cancellation_handler=None,
        token_callback=None,
    ) -> Tuple[Dict, Dict, Optional[str]]:
        """
        Translates specified fields within a frontmatter dictionary into a target language.

        The method processes the given fields, sends them for translation using the specified model, and updates the frontmatter copy with translated values. Supports streaming responses, cancellation, and token callbacks. If an error occurs, returns the original frontmatter data.

        Args:
            frontmatter_data: Dictionary containing the frontmatter to translate.
            fields: List of field names within the frontmatter to translate.
            target_language: Language to translate the fields into.
            model: Model identifier to use for translation.
            stream: If True, enables streaming of translation tokens.
            cancellation_handler: Optional handler to support cancellation during streaming.
            token_callback: Optional function called with each token during streaming.

        Returns:
            A tuple containing:
                - The frontmatter dictionary with translated fields (original if error occurs).
                - A dictionary with token usage statistics.
                - An error message string if an error occurred, otherwise None.
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
            provider = ProviderFactory.create_provider(
                model,
                openai_client=self.openai_client,
                anthropic_client=self.anthropic_client
            )

            translated_text, usage, error = provider.translate_text(
                text=fields_text,
                target_language=target_language,
                model=model,
                system_prompt=system_prompt,
                stream=stream,
                cancellation_handler=cancellation_handler,
                token_callback=token_callback
            )

            if translated_text:
                # Log the frontmatter translation prompts and response
                self.translation_log["frontmatter"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": translated_text,
                    "usage": usage,
                    "fields": fields,
                    "streaming": stream,
                }

                # Extract each translated field from the response
                for field in fields:
                    pattern = rf"{field}: (.*?)(?:\n\n|\n$|$)"
                    match = re.search(pattern, translated_text, re.DOTALL)
                    if match:
                        translated_value = match.group(1).strip()
                        translated_frontmatter[field] = translated_value

                return translated_frontmatter, usage, None
            else:
                return frontmatter_data, empty_usage, error

        except Exception as e:
            error_msg = f"Failed to translate frontmatter: {str(e)}"
            # Return original frontmatter on error
            return frontmatter_data, empty_usage, error_msg
