#!/usr/bin/env python3
# ABOUTME: File input/output utilities for the translator.
# ABOUTME: Provides functions to read, write, and generate output filenames.

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markup import escape

from translator.language import LanguageHandler

console = Console()


class FileHandler:
    """File input/output utilities for the translator."""

    @staticmethod
    def read_file(file_path: str) -> str:
        """Read content from a file.
        
        Args:
            file_path: The path to the file to read
            
        Returns:
            The content of the file as a string
            
        Raises:
            SystemExit: If the file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to read file: {escape(str(e))}")
            sys.exit(1)

    @staticmethod
    def write_file(file_path: str, content: str) -> None:
        """Write content to a file.
        
        Args:
            file_path: The path to the file to write
            content: The content to write to the file
            
        Raises:
            SystemExit: If the file cannot be written
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to write file: {escape(str(e))}")
            sys.exit(1)

    @staticmethod
    def write_log(log_path: str, log_data: dict) -> None:
        """Write detailed translation log to a file.
        
        Args:
            log_path: The path to the log file
            log_data: Dictionary containing the log data
            
        Raises:
            SystemExit: If the log file cannot be written
        """
        try:
            import json
            from datetime import datetime
            
            # Add timestamp to the log
            log_data["timestamp"] = datetime.now().isoformat()
            
            # Format the log content
            log_content = json.dumps(log_data, indent=2, ensure_ascii=False)
            
            with open(log_path, 'w', encoding='utf-8') as file:
                file.write(log_content)
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Failed to write log file: {escape(str(e))}")
            # Don't exit on log failure, just warn

    @staticmethod
    def get_output_filename(input_file: str, target_language: str, output_file: Optional[str] = None) -> str:
        """Generate output filename if not provided.
        
        Args:
            input_file: The path to the input file
            target_language: The target language for translation
            output_file: Optional custom output file path
            
        Returns:
            The path to the output file
        """
        if output_file:
            return output_file
        
        # Get language code
        language_code = LanguageHandler.get_language_code(target_language)
        
        input_path = Path(input_file)
        parent_dir = input_path.parent
        stem = input_path.stem
        suffix = input_path.suffix
        
        # Create a new path in the same directory as the input file
        return str(parent_dir / f"{stem}.{language_code}{suffix}")
        
    @staticmethod
    def get_log_filename(output_file: str) -> str:
        """Generate log filename based on the output file.
        
        Args:
            output_file: The path to the output file
            
        Returns:
            The path to the log file
        """
        output_path = Path(output_file)
        return str(output_path.with_suffix(f"{output_path.suffix}.log.json"))