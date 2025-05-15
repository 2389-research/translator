#!/usr/bin/env python3
# ABOUTME: Frontmatter parsing and handling for markdown files.
# ABOUTME: Processes YAML frontmatter in blog posts and static site content.

from typing import Dict, List, Optional, Tuple

import frontmatter
from rich.console import Console
from rich.markup import escape

console = Console()


class FrontmatterHandler:
    """Frontmatter parsing and handling for markdown files."""

    @staticmethod
    def parse_frontmatter(content: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Parse frontmatter from content using python-frontmatter.

        Args:
            content: The content to parse

        Returns:
            Tuple containing:
                - Boolean indicating if frontmatter was detected
                - Dictionary containing the frontmatter data if found, otherwise None
                - String containing the content without frontmatter if found, otherwise None
        """
        try:
            # Parse content with frontmatter
            post = frontmatter.loads(content)

            # Check if frontmatter was found
            if post.metadata:
                # Extract metadata and content
                metadata = dict(post.metadata)
                content_without_frontmatter = post.content
                return True, metadata, content_without_frontmatter
            else:
                # No frontmatter found
                return False, None, None
        except Exception as e:
            console.print(
                f"[bold yellow]Warning:[/] Failed to parse frontmatter: {escape(str(e))}"
            )
            return False, None, None

    @staticmethod
    def get_translatable_frontmatter_fields(frontmatter_data: Dict) -> List[str]:
        """Get a list of frontmatter fields that should be translated.

        Args:
            frontmatter_data: The frontmatter data as a dictionary

        Returns:
            A list of field names that should be translated
        """
        # Common translatable fields in various static site generators
        translatable_fields = [
            "title",
            "description",
            "summary",
            "excerpt",
            "subtitle",
            "seo_title",
            "seo_description",
            "meta_description",
            "abstract",
            "intro",
            "heading",
            "subheading",
        ]

        # Return only fields that exist in the frontmatter
        return [field for field in translatable_fields if field in frontmatter_data]

    @staticmethod
    def reconstruct_with_frontmatter(metadata: Dict, content: str) -> str:
        """Reconstruct content with frontmatter using python-frontmatter.

        Args:
            metadata: The frontmatter metadata as a dictionary
            content: The content to include with the frontmatter

        Returns:
            The reconstructed content with frontmatter
        """
        # Create a new post object with metadata and content
        post = frontmatter.Post(content, **metadata)

        # Return the serialized post
        return frontmatter.dumps(post)
