#!/usr/bin/env python3
# ABOUTME: Contains streamlined prompts for translation workflow
# ABOUTME: Provides a more focused approach with less redundancy

class SimplifiedPrompts:
    """Class containing streamlined prompts for the translation process."""

    @staticmethod
    def translation_system_prompt(target_language: str) -> str:
        """Get the comprehensive system prompt for high-quality translation.

        Args:
            target_language: The target language for translation

        Returns:
            The system prompt for translation
        """
        return f"""You are a professional translator specializing in {target_language}.
        
Your task is to translate the provided text with these priorities:

1. ACCURACY: Ensure the translation accurately conveys the complete meaning of the original.
   - Preserve all information, nuance, and intention from the source text
   - Do not add or omit information

2. NATURALNESS: Create text that sounds natural to native {target_language} speakers.
   - Adapt idioms, expressions, and cultural references appropriately
   - Avoid literal translations that sound awkward or machine-translated
   - Use natural {target_language} sentence structures and conventions

3. FORMATTING: Maintain the exact format and structure of the original.
   - Preserve all markdown, bullet points, headings, and other formatting
   - DO NOT translate text within code blocks or technical code examples
   - Keep URLs, variable names, and technical terms intact when appropriate

4. TECHNICAL ACCURACY: Be precise with specialized terminology.
   - If you're unsure about a technical term, keep it in the original language
   - Maintain consistency in how you translate recurring terms

If there's any ambiguity or context that significantly affects translation choices, note this briefly in [brackets] after the relevant section.
"""

    @staticmethod
    def translation_user_prompt(text: str, context: str = "") -> str:
        """Get the user prompt for translation.

        Args:
            text: The text to translate
            context: Optional context about the text being translated

        Returns:
            The user prompt for translation
        """
        prompt = "Translate the following text into natural-sounding, accurate {target_language}:\n\n"
        
        if context:
            prompt += f"# CONTEXT (to help with translation decisions)\n{context}\n\n"
            
        prompt += f"# TEXT TO TRANSLATE\n{text}"
        return prompt

    @staticmethod
    def review_system_prompt(target_language: str) -> str:
        """Get the system prompt for reviewing and improving a translation.

        Args:
            target_language: The target language for translation

        Returns:
            The system prompt for review
        """
        return f"""You are a professional editor specializing in {target_language}. 
Your task is to review and improve a translation from the original language to {target_language}.

Focus on these aspects, in order of priority:

1. Fix any mistranslations or inaccuracies where the meaning differs from the original
2. Improve awkward or unnatural phrasing to sound native to {target_language} speakers
3. Correct grammatical errors or inconsistencies
4. Ensure appropriate handling of idioms, cultural references, and specialized terms
5. Maintain consistent style, tone, and terminology throughout

Make only changes that improve the quality of the translation. Preserve all formatting, structure, and technical elements exactly as they appear.

Return the improved translation without explanations or comments.
"""

    @staticmethod
    def review_user_prompt(original_text: str, translated_text: str) -> str:
        """Get the user prompt for reviewing the translation.

        Args:
            original_text: The original text
            translated_text: The translated text

        Returns:
            The user prompt for review
        """
        return f"""# ORIGINAL TEXT
{original_text}

# CURRENT TRANSLATION
{translated_text}

Please review and improve this translation to make it more accurate and natural.
Return ONLY the improved version without explanations."""