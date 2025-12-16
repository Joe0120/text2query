"""JSON parsing utilities for handling LLM responses."""

from __future__ import annotations

import ast
import json
import re
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def clean_json_string(json_str: str) -> str:
    """
    Clean JSON string by removing comments and fixing common issues.

    Args:
        json_str: Raw JSON string that may contain comments or formatting issues

    Returns:
        Cleaned JSON string
    """
    original = json_str

    try:
        # 1. Remove single-line comments (// ...)
        lines = json_str.split('\n')
        cleaned_lines = []

        for line in lines:
            in_string = False
            escape_next = False
            comment_start = -1

            for i, char in enumerate(line):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string and i < len(line) - 1:
                    if line[i:i+2] == '//':
                        comment_start = i
                        break

            if comment_start >= 0:
                line = line[:comment_start].rstrip()

            if line.strip():
                cleaned_lines.append(line)

        json_str = '\n'.join(cleaned_lines)

        # 2. Remove multi-line comments (/* ... */)
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)

        # 3. Fix trailing comma issues
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        # 4. Remove BOM and invisible characters
        json_str = json_str.replace('\ufeff', '')
        json_str = json_str.replace('\u200b', '')

        # 5. Fix unclosed brackets
        open_brackets = json_str.count('[') - json_str.count(']')
        open_braces = json_str.count('{') - json_str.count('}')

        if open_brackets > 0:
            json_str += ']' * open_brackets
        if open_braces > 0:
            json_str += '}' * open_braces

        return json_str.strip()

    except Exception:
        return original


def extract_json_from_string(resp_content: str) -> Optional[Union[Dict, List]]:
    """
    Extract JSON from a string, supporting various formats and fixing incomplete JSON.

    Args:
        resp_content: Response content that may contain JSON

    Returns:
        Parsed JSON as dict or list, or None if parsing fails
    """
    if not resp_content:
        return None

    content = resp_content

    # Patterns to extract JSON content
    json_patterns = [
        r'```json\s*([\s\S]*?)```',
        r'```json\=\s*([\s\S]*?)```',
        r'```\s*([\{\[][\s\S]*?[\]\}])\s*```',
        r'([\{\[][\s\S]*[\]\}])',
    ]

    match_str = ''
    for pattern in json_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            match_str = match.group(1)
            break

    if not match_str:
        match_str = content

    cleaned_json = clean_json_string(match_str)

    # Try parsing methods in order of preference
    parse_methods = [
        ('json.loads', lambda s: json.loads(s)),
        ('ast.literal_eval', lambda s: ast.literal_eval(s)),
    ]

    for method_name, method_func in parse_methods:
        try:
            result = method_func(cleaned_json)
            return result
        except Exception:
            continue

    # Try original string if cleaned version fails
    if match_str != cleaned_json:
        for method_name, method_func in parse_methods:
            try:
                result = method_func(match_str)
                return result
            except Exception:
                continue

    return None


def extract_list_from_string(resp_content: str) -> Optional[List]:
    """
    Extract a list from a string response.

    Args:
        resp_content: Response content that may contain a list

    Returns:
        Parsed list, or None if parsing fails
    """
    resp_content = resp_content.replace('\n', '')
    match_str = ''

    if re.search(r'```([^\[]*)\[', resp_content):
        resp_content = re.sub(r'```([^\[]*)\[', '```[', resp_content)
    if re.search(r'],([^\[\]]*)', resp_content):
        resp_content = re.sub(r'],([^\[\]]*)', '],', resp_content)

    if re.search(r'```([\[].*[\]])```', resp_content):
        match_str = re.search(r'```([\[].*[\]])```', resp_content).group(1)
    elif re.search(r'([\[].*[\]])', resp_content):
        match_str = re.search(r'([\[].*[\]])', resp_content).group(1)

    if match_str:
        try:
            return list(ast.literal_eval(match_str))
        except Exception as e:
            logger.warning(f"Failed to parse list: {e}")
            return None

    return None


def convert_to_dict(data: Any) -> Any:
    """
    Recursively convert special dict types to regular dictionaries.

    Args:
        data: Data that may contain special dict types

    Returns:
        Data with all dicts converted to regular dicts
    """
    if isinstance(data, dict):
        return {key: convert_to_dict(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_dict(item) for item in data]
    else:
        return data
