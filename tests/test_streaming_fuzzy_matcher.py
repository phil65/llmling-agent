"""Test suite for the streaming fuzzy matcher implementation.

This module contains comprehensive tests for the StreamingFuzzyMatcher class,
covering various scenarios including exact matches, fuzzy matches, streaming
behavior, and edge cases.
"""

import pytest

from agentpool_toolsets.builtin.file_edit.fuzzy_matcher import (
    Range,
    StreamingFuzzyMatcher,
)


def test_empty_query():
    """Test matcher with empty query returns no matches."""
    source = "def hello_world():\n    print('Hello, World!')\n"
    matcher = StreamingFuzzyMatcher(source)

    result = matcher.finish()
    assert result == []


def test_streaming_exact_match():
    """Test exact match with streaming input."""
    source = "def hello_world():\n    print('Hello, World!')\n"
    matcher = StreamingFuzzyMatcher(source)

    # Push query in chunks
    result1 = matcher.push("def hello")
    assert result1 is None  # No complete line yet

    result2 = matcher.push("_world():\n")
    assert result2 is not None
    assert isinstance(result2, Range)

    # Verify the match covers the right area
    matched_text = source[result2.start : result2.end]
    assert "def hello_world():" in matched_text


def test_streaming_fuzzy_match():
    """Test fuzzy matching with typos."""
    source = """def calculate_sum(a, b):
return a + b

def calculate_product(a, b):
return a * b
"""
    matcher = StreamingFuzzyMatcher(source)

    # Query with typo: "calcuate" instead of "calculate"
    result = matcher.push("def calcuate_sum(a, b):\n")
    assert result is not None

    matched_text = source[result.start : result.end]
    assert "calculate_sum" in matched_text


def test_incremental_improvement():
    """Test that matches improve as more query text is added."""
    source = """class Calculator:
def add(self, x, y):
    return x + y

def multiply(self, x, y):
    return x * y
"""
    matcher = StreamingFuzzyMatcher(source)

    # Start with ambiguous query
    _result1 = matcher.push("def ")
    # Should be None or very broad match

    # Add more specific info
    result2 = matcher.push("add(self, x, y):\n")
    assert result2 is not None

    matched_text = source[result2.start : result2.end]
    assert "add" in matched_text


def test_incomplete_lines_buffering():
    """Test that incomplete lines are properly buffered."""
    source = "def test_function():\n    pass\n"
    matcher = StreamingFuzzyMatcher(source)

    # Push partial line
    result1 = matcher.push("def test")
    assert result1 is None

    result2 = matcher.push("_func")
    assert result2 is None

    # Complete the line
    result3 = matcher.push("tion():\n")
    assert result3 is not None


def test_multiline_fuzzy_match():
    """Test fuzzy matching across multiple lines."""
    source = """import os
import sys

def main():
print("Hello World")
return 0

if __name__ == "__main__":
main()
"""
    matcher = StreamingFuzzyMatcher(source)

    # Multi-line query with slight variations
    matcher.push("import os\n")
    matcher.push("import sys\n")
    result = matcher.push("def main():\n")

    assert result is not None
    matched_text = source[result.start : result.end]
    assert "import os" in matched_text
    assert "def main():" in matched_text


def test_resolve_location_single_line():
    """Test location resolution for single line match."""
    source = "x = 42\ny = 24\nz = x + y\n"
    matcher = StreamingFuzzyMatcher(source)

    result = matcher.push("y = 24\n")
    assert result is not None

    matched_text = source[result.start : result.end]
    assert "y = 24" in matched_text


def test_resolve_location_multiline():
    """Test location resolution for multiline match."""
    source = """def fibonacci(n):
if n <= 1:
    return n
return fibonacci(n-1) + fibonacci(n-2)
"""
    matcher = StreamingFuzzyMatcher(source)

    matcher.push("def fibonacci(n):\n")
    result = matcher.push("    if n <= 1:\n")

    assert result is not None
    matched_text = source[result.start : result.end]
    assert "fibonacci" in matched_text
    assert "if n <= 1:" in matched_text


def test_resolve_location_function_with_typo():
    """Test resolving function location with typos."""
    source = """def process_data(input_data):
cleaned = input_data.strip()
return cleaned.upper()
"""
    matcher = StreamingFuzzyMatcher(source)

    # Query with typo: "proces" instead of "process"
    result = matcher.push("def proces_data(input_data):\n")
    assert result is not None

    matched_text = source[result.start : result.end]
    assert "process_data" in matched_text


def test_resolve_location_class_methods():
    """Test resolving class method locations."""
    source = """class DataProcessor:
def __init__():
    self.data = []

def add_item(self, item):
    self.data.append(item)

def process_all():
    return [item.upper() for item in self.data]
"""
    matcher = StreamingFuzzyMatcher(source)

    result = matcher.push("    def add_item(self, item):\n")
    assert result is not None

    matched_text = source[result.start : result.end]
    assert "add_item" in matched_text


def test_resolve_location_imports_no_match():
    """Test that non-matching imports return no match."""
    source = """import json
import requests
from typing import List, Dict

def api_call():
pass
"""
    matcher = StreamingFuzzyMatcher(source)

    # Query for import that doesn't exist
    result = matcher.push("import pandas\n")
    # Should either be None or a very poor match
    if result is not None:
        matched_text = source[result.start : result.end]
        # Should not be a high-confidence match
        assert len(matched_text) < len(source) // 2


def test_resolve_location_nested_closure():
    """Test resolving nested function locations."""
    source = """def outer_function():
def inner_function():
    return "nested"

result = inner_function()
return result
"""
    matcher = StreamingFuzzyMatcher(source)

    result = matcher.push("    def inner_function():\n")
    assert result is not None

    matched_text = source[result.start : result.end]
    assert "inner_function" in matched_text


def test_resolve_location_tool_invocation():
    """Test resolving tool invocation patterns."""
    source = """def execute_command(cmd):
import subprocess
result = subprocess.run(cmd, shell=True)
return result.returncode
"""
    matcher = StreamingFuzzyMatcher(source)

    matcher.push("def execute_command(cmd):\n")
    result = matcher.push("    import subprocess\n")

    assert result is not None
    matched_text = source[result.start : result.end]
    assert "execute_command" in matched_text
    assert "subprocess" in matched_text


def test_line_hint_selection():
    """Test line hint affects match selection."""
    source = """# Line 1
def function_one():
pass

# Line 5
def function_two():
pass

# Line 9
def function_three():
pass
"""
    matcher = StreamingFuzzyMatcher(source)

    # Without line hint - should find first match
    result1 = matcher.push("def function")
    matcher_copy = StreamingFuzzyMatcher(source)

    # With line hint pointing to second function
    result2 = matcher_copy.push("def function", line_hint=5)

    if result1 and result2:
        # Results should be different
        text1 = source[result1.start : result1.end]
        text2 = source[result2.start : result2.end]
        # At least one should contain different function names
        assert text1 != text2 or "function_two" in text2


def test_finish_processes_incomplete_line():
    """Test that finish() processes remaining incomplete lines."""
    source = "def test():\n    return True\n"
    matcher = StreamingFuzzyMatcher(source)

    # Push without newline
    matcher.push("def test():")

    # Finish should process the incomplete line
    results = matcher.finish()
    assert len(results) >= 1

    matched_text = source[results[0].start : results[0].end]
    assert "def test():" in matched_text


def test_multiple_matches_ambiguous():
    """Test behavior with multiple ambiguous matches."""
    source = """def helper():
pass

def helper_function():
pass

def another_helper():
pass
"""
    matcher = StreamingFuzzyMatcher(source)

    # Ambiguous query matching multiple functions
    result = matcher.push("def helper\n")

    # Should either pick one or return None due to ambiguity
    if result is not None:
        matched_text = source[result.start : result.end]
        assert "helper" in matched_text


def test_empty_source_text():
    """Test matcher with empty source text."""
    matcher = StreamingFuzzyMatcher("")

    result = matcher.push("some query\n")
    assert result is None

    results = matcher.finish()
    assert results == []


def test_query_longer_than_source():
    """Test when query is longer than source text."""
    source = "short\n"
    matcher = StreamingFuzzyMatcher(source)

    long_query = "this is a very long query that is much longer than the source\n"
    result = matcher.push(long_query)

    # Should handle gracefully
    assert result is None or isinstance(result, Range)


def test_special_characters_and_unicode():
    """Test matching with special characters and unicode."""
    source = """def café_function():
return "Hello 世界!"

def test_symbols():
return "@#$%^&*()"
"""
    matcher = StreamingFuzzyMatcher(source)

    result = matcher.push("def café_function():\n")
    assert result is not None

    matched_text = source[result.start : result.end]
    assert "café_function" in matched_text


def test_whitespace_sensitivity():
    """Test how matcher handles different whitespace patterns."""
    source = """def function_with_spaces( a , b ):
return a+b

def function_no_spaces(a,b):
return a+b
"""
    matcher = StreamingFuzzyMatcher(source)

    # Query with different whitespace
    result = matcher.push("def function_with_spaces(a, b):\n")
    assert result is not None

    matched_text = source[result.start : result.end]
    assert "function_with_spaces" in matched_text


def test_range_class():
    """Test Range dataclass functionality."""
    range_obj = Range(10, 20)
    assert range_obj.start == 10  # noqa: PLR2004
    assert range_obj.end == 20  # noqa: PLR2004

    # Test immutability
    with pytest.raises(Exception):  # Should be frozen  # noqa: B017, PT011
        range_obj.start = 15  # type: ignore


def test_fuzzy_equality():
    """Test the fuzzy string equality function indirectly."""
    source = "def calculate_result():\n    pass\n"
    matcher = StreamingFuzzyMatcher(source)

    # Test that similar strings match
    result1 = matcher.push("def calculate_result():\n")
    assert result1 is not None

    # Test that very different strings don't match well
    matcher2 = StreamingFuzzyMatcher(source)
    result2 = matcher2.push("class MyClass:\n")

    # Should either be None or a poor match
    if result1 and result2:
        # The exact match should be better (shorter range or better position)
        assert result1.start <= result2.start or (result1.end - result1.start) <= (
            result2.end - result2.start
        )


if __name__ == "__main__":
    pytest.main([__file__])
