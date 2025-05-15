#!/usr/bin/env python3
# ABOUTME: Contains the prompts used for translation, editing, and critique.
# ABOUTME: Provides a centralized location for all prompts used in the translation process.


class Prompts:
    """Class containing all prompts used in the translation process."""

    @staticmethod
    def translation_system_prompt(target_language: str) -> str:
        """Get the system prompt for translation.

        Args:
            target_language: The target language for translation

        Returns:
            The system prompt for translation
        """
        return f"""
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

    @staticmethod
    def translation_user_prompt(text: str) -> str:
        """Get the user prompt for translation.

        Args:
            text: The text to translate

        Returns:
            The user prompt for translation
        """
        return text

    @staticmethod
    def editing_system_prompt(target_language: str) -> str:
        """Get the system prompt for editing the translation.

        Args:
            target_language: The target language for translation

        Returns:
            The system prompt for editing
        """
        return f"""
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

    @staticmethod
    def editing_user_prompt(
        original_text: str, translated_text: str, target_language: str
    ) -> str:
        """Get the user prompt for editing the translation.

        Args:
            original_text: The original text
            translated_text: The translated text
            target_language: The target language

        Returns:
            The user prompt for editing
        """
        return f"""# ORIGINAL TEXT
{original_text}

# TRANSLATED TEXT
{translated_text}

Please review and improve the translated text to make it natural and accurate in {target_language}.
Return ONLY the improved translated text without explanations or comments."""

    @staticmethod
    def critique_system_prompt(target_language: str) -> str:
        """Get the system prompt for critiquing the translation.

        Args:
            target_language: The target language for translation

        Returns:
            The system prompt for critique
        """
        return f"""You are a highly critical professional translator and linguistic expert specializing in {target_language}.
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

    @staticmethod
    def critique_user_prompt(original_text: str, translated_text: str) -> str:
        """Get the user prompt for critiquing the translation.

        Args:
            original_text: The original text
            translated_text: The translated text

        Returns:
            The user prompt for critique
        """
        return f"""# ORIGINAL TEXT
{original_text}

# CURRENT TRANSLATION
{translated_text}

Please critique this translation mercilessly and provide detailed feedback on what needs to be improved.
Format your critique as a structured list of issues, organized by severity and category.
Include specific suggestions for how to fix each issue."""

    @staticmethod
    def feedback_system_prompt(target_language: str) -> str:
        """Get the system prompt for applying critique feedback.

        Args:
            target_language: The target language for translation

        Returns:
            The system prompt for applying feedback
        """
        return f"""You are a master translator and editor specializing in {target_language}.
Your task is to improve a translation based on detailed critique feedback.

1. Carefully read the original text, current translation, and the critique feedback
2. Address ALL issues identified in the critique
3. Apply the specific suggestions for improvement
4. Ensure the translation is accurate, natural-sounding, and faithful to the original
5. Preserve all formatting, markdown, and structure of the original text
6. Make sure the final text reads as if it were originally written in {target_language}

Do not ignore any of the critique points. Every issue identified must be addressed in your improved version.
"""

    @staticmethod
    def feedback_user_prompt(
        original_text: str, translated_text: str, critique_feedback: str
    ) -> str:
        """Get the user prompt for applying critique feedback.

        Args:
            original_text: The original text
            translated_text: The translated text
            critique_feedback: The critique feedback

        Returns:
            The user prompt for applying feedback
        """
        return f"""# ORIGINAL TEXT
{original_text}

# CURRENT TRANSLATION
{translated_text}

# CRITIQUE FEEDBACK
{critique_feedback}

Please address ALL issues identified in the critique and provide an improved translation.
Return ONLY the improved translated text without explanations or comments."""

    @staticmethod
    def frontmatter_system_prompt(target_language: str) -> str:
        """Get the system prompt for translating frontmatter.

        Args:
            target_language: The target language for translation

        Returns:
            The system prompt for frontmatter translation
        """
        return f"""You are a professional translator. Translate the following frontmatter fields to {target_language}.
Each field is in the format "field_name: content". Translate ONLY the content, not the field names.
Return the translated content in the exact same format, preserving all field names."""

    @staticmethod
    def frontmatter_user_prompt(fields_text: str) -> str:
        """Get the user prompt for translating frontmatter.

        Args:
            fields_text: The text containing frontmatter fields to translate

        Returns:
            The user prompt for frontmatter translation
        """
        return fields_text
