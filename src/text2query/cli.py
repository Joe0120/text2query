"""
Command line interface for text2query
"""

import argparse
import sys
from typing import Optional

from .core.legacy_core import Text2Query
from .exceptions import Text2QueryError


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="Convert text to database queries",
        prog="text2query"
    )
    
    parser.add_argument(
        "text",
        help="Text to convert to query"
    )
    
    parser.add_argument(
        "-t", "--type",
        choices=["sql", "mongodb"],
        default="sql",
        help="Query type to generate (default: sql)"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="text2query 0.0.1"
    )
    
    args = parser.parse_args()
    
    try:
        converter = Text2Query()
        result = converter.convert(args.text, args.type)
        print(result)
        
    except Text2QueryError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
