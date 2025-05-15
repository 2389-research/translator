#!/usr/bin/env python3
# ABOUTME: Core translation logic using OpenAI's API.
# ABOUTME: Provides translation, editing, and critique functions.

import re
from typing import Dict, List, Optional, Tuple

import openai

from translator.prompts import Prompts


class Translator:
    """Core translation logic using OpenAI's API.
    
    This class provides methods for translating text, editing translations, 
    critiquing translations, and applying critique feedback. All methods support 
    both streaming and non-streaming responses from the OpenAI API.
    
    Streaming responses provide the following benefits:
    1. Lower latency for displaying initial results
    2. More responsive user experience
    3. Possibility of cancelling long-running requests
    """

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
        self, text: str, target_language: str, model: str, stream: bool = False,
        cancellation_handler=None
    ) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Translate text to the target language using OpenAI.

        Args:
            text: The text to translate
            target_language: The target language for translation
            model: The model to use for translation
            stream: Whether to stream responses (default: False)
            cancellation_handler: Optional handler to check for cancellation requests

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
                stream=stream
            )

            # Handle streaming response
            if stream:
                # For streaming, we collect tokens and manually track usage
                full_response = ""
                # Initialize estimated usage with empty values
                usage = {
                    "prompt_tokens": 0,  # Will estimate later
                    "completion_tokens": 0,  # Will count during streaming
                    "total_tokens": 0,  # Will calculate at the end
                }
                
                for chunk in response:
                    # Check for cancellation if handler is provided
                    if cancellation_handler and cancellation_handler.is_cancellation_requested():
                        break
                        
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Count tokens in chunk.choices[0].delta.content
                        usage["completion_tokens"] += 1  # This is a rough estimate
                
                # Estimate prompt tokens based on input length
                from translator.token_counter import TokenCounter
                prompt_str = system_prompt + user_prompt
                usage["prompt_tokens"] = TokenCounter.count_tokens(prompt_str, model)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                
                # Log the translation prompts and response
                self.translation_log["translation"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": full_response,
                    "usage": usage,
                    "streaming": True,
                }
                
                return full_response, usage, None
            else:
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
                    "streaming": False,
                }

                return response.choices[0].message.content, usage, None
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            return None, empty_usage, error_msg

    def edit_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str,
        stream: bool = False, cancellation_handler=None
    ) -> Tuple[str, Dict, Optional[str]]:
        """Edit the translation to ensure it makes sense in the target language while preserving original meaning.

        Args:
            translated_text: The translated text to edit
            original_text: The original text for reference
            target_language: The target language
            model: The model to use for editing
            stream: Whether to stream responses (default: False)
            cancellation_handler: Optional handler to check for cancellation requests

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
                
            # Add stream parameter if requested
            if stream:
                params["stream"] = True
            
            response = self.client.chat.completions.create(**params)
            
            # Handle streaming response
            if stream:
                # For streaming, we collect tokens and manually track usage
                full_response = ""
                # Initialize estimated usage with empty values
                usage = {
                    "prompt_tokens": 0,  # Will estimate later
                    "completion_tokens": 0,  # Will count during streaming
                    "total_tokens": 0,  # Will calculate at the end
                }
                
                for chunk in response:
                    # Check for cancellation if handler is provided
                    if cancellation_handler and cancellation_handler.is_cancellation_requested():
                        break
                        
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Count tokens in chunk.choices[0].delta.content
                        usage["completion_tokens"] += 1  # This is a rough estimate
                
                # Estimate prompt tokens based on input length
                from translator.token_counter import TokenCounter
                # Estimate prompt tokens by combining system and user prompts
                prompt_str = system_prompt + user_prompt
                usage["prompt_tokens"] = TokenCounter.count_tokens(prompt_str, model)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                
                # Log the editing prompts and response
                self.translation_log["editing"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": full_response,
                    "usage": usage,
                    "streaming": True,
                }
                
                return full_response, usage, None
            else:
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
                    "streaming": False,
                }

                return response.choices[0].message.content, usage, None
        except Exception as e:
            error_msg = f"Editing failed: {str(e)}"
            # Return original translation if editing fails with empty usage stats
            return translated_text, empty_usage, error_msg

    def critique_translation(
        self, translated_text: str, original_text: str, target_language: str, model: str,
        stream: bool = False, cancellation_handler=None
    ) -> Tuple[str, Dict, str, Optional[str]]:
        """Aggressively critique the translation against the original text.

        Args:
            translated_text: The translated text to critique
            original_text: The original text for reference
            target_language: The target language
            model: The model to use for critique
            stream: Whether to stream responses (default: False)
            cancellation_handler: Optional handler to check for cancellation requests

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
                
            # Add stream parameter if requested
            if stream:
                params["stream"] = True

            response = self.client.chat.completions.create(**params)
            
            # Handle streaming response
            if stream:
                # For streaming, we collect tokens and manually track usage
                full_response = ""
                # Initialize estimated usage with empty values
                usage = {
                    "prompt_tokens": 0,  # Will estimate later
                    "completion_tokens": 0,  # Will count during streaming
                    "total_tokens": 0,  # Will calculate at the end
                }
                
                for chunk in response:
                    # Check for cancellation if handler is provided
                    if cancellation_handler and cancellation_handler.is_cancellation_requested():
                        break
                        
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Count tokens in chunk.choices[0].delta.content
                        usage["completion_tokens"] += 1  # This is a rough estimate
                
                # Estimate prompt tokens based on input length
                from translator.token_counter import TokenCounter
                # Estimate prompt tokens by combining system and user prompts
                prompt_str = system_prompt + user_prompt
                usage["prompt_tokens"] = TokenCounter.count_tokens(prompt_str, model)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                
                # Log the critique prompts and response
                self.translation_log["critique"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": full_response,
                    "usage": usage,
                    "streaming": True,
                }
                
                critique_feedback = full_response
                return translated_text, usage, critique_feedback, None
            else:
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
                    "streaming": False,
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
        stream: bool = False,
        cancellation_handler=None,
    ) -> Tuple[str, Dict, Optional[str]]:
        """Apply critique feedback to improve the translation.

        Args:
            translated_text: The translated text to improve
            original_text: The original text for reference
            critique_feedback: The detailed critique feedback
            target_language: The target language
            model: The model to use for applying feedback
            stream: Whether to stream responses (default: False)
            cancellation_handler: Optional handler to check for cancellation requests

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
                
            # Add stream parameter if requested
            if stream:
                params["stream"] = True

            response = self.client.chat.completions.create(**params)
            
            # Handle streaming response
            if stream:
                # For streaming, we collect tokens and manually track usage
                full_response = ""
                # Initialize estimated usage with empty values
                usage = {
                    "prompt_tokens": 0,  # Will estimate later
                    "completion_tokens": 0,  # Will count during streaming
                    "total_tokens": 0,  # Will calculate at the end
                }
                
                for chunk in response:
                    # Check for cancellation if handler is provided
                    if cancellation_handler and cancellation_handler.is_cancellation_requested():
                        break
                        
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Count tokens in chunk.choices[0].delta.content
                        usage["completion_tokens"] += 1  # This is a rough estimate
                
                # Estimate prompt tokens based on input length
                from translator.token_counter import TokenCounter
                # Estimate prompt tokens by combining system and user prompts
                prompt_str = system_prompt + user_prompt
                usage["prompt_tokens"] = TokenCounter.count_tokens(prompt_str, model)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                
                # Log the feedback application prompts and response
                self.translation_log["feedback"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": full_response,
                    "usage": usage,
                    "streaming": True,
                }
                
                return full_response, usage, None
            else:
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
                    "streaming": False,
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
        stream: bool = False,
        cancellation_handler=None,
    ) -> Tuple[Dict, Dict, Optional[str]]:
        """Translate specified fields in frontmatter.

        Args:
            frontmatter_data: The frontmatter data as a dictionary
            fields: List of fields to translate
            target_language: The target language
            model: The model to use for translation
            stream: Whether to stream responses (default: False)
            cancellation_handler: Optional handler to check for cancellation requests

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
                stream=stream
            )

            # Handle streaming response
            if stream:
                # For streaming, we collect tokens and manually track usage
                full_response = ""
                # Initialize estimated usage with empty values
                usage = {
                    "prompt_tokens": 0,  # Will estimate later
                    "completion_tokens": 0,  # Will count during streaming
                    "total_tokens": 0,  # Will calculate at the end
                }
                
                for chunk in response:
                    # Check for cancellation if handler is provided
                    if cancellation_handler and cancellation_handler.is_cancellation_requested():
                        break
                        
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Count tokens in chunk.choices[0].delta.content
                        usage["completion_tokens"] += 1  # This is a rough estimate
                
                # Estimate prompt tokens based on input length
                from translator.token_counter import TokenCounter
                # Estimate prompt tokens by combining system and user prompts
                prompt_str = system_prompt + user_prompt
                usage["prompt_tokens"] = TokenCounter.count_tokens(prompt_str, model)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                
                # Log the frontmatter translation prompts and response
                self.translation_log["frontmatter"] = {
                    "model": model,
                    "target_language": target_language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": full_response,
                    "usage": usage,
                    "fields": fields,
                    "streaming": True,
                }
                
                translated_text = full_response
            else:
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
                    "streaming": False,
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
