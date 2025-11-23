"""Example usage of the StreamingFuzzyMatcher.

This script demonstrates how to use the streaming fuzzy matcher for
real-time code location resolution, simulating how it might be used
in a code editor or AI assistant.
"""

import time

from llmling_agent_toolsets.builtin.file_edit.fuzzy_matcher import StreamingFuzzyMatcher


def simulate_streaming_input(
    matcher: StreamingFuzzyMatcher, query: str, chunk_size: int = 5
) -> None:
    """Simulate streaming input by sending query in small chunks.

    Args:
        matcher: The fuzzy matcher instance
        query: Query text to send in chunks
        chunk_size: Size of each chunk to send
    """
    print(f"🔍 Streaming query: {query!r}")
    print("─" * 50)

    for i in range(0, len(query), chunk_size):
        chunk = query[i : i + chunk_size]
        result = matcher.push(chunk)

        print(f"📨 Pushed chunk: {chunk!r}")
        if result:
            print(f"✅ Found match: Range({result.start}, {result.end})")
            # Show matched text preview
            source_lines = "\n".join(matcher.source_lines)
            matched_text = source_lines[result.start : result.end]
            preview = matched_text.replace("\n", "\\n")[:60]
            if len(matched_text) > 60:  # noqa: PLR2004
                preview += "..."
            print(f"📝 Preview: {preview!r}")
        else:
            print("⏳ No match yet...")

        print()
        time.sleep(0.1)  # Simulate real-time delay


def demonstrate_exact_matching() -> None:
    """Demo exact matching with streaming input."""
    print("🎯 EXACT MATCHING DEMO")
    print("=" * 60)

    source_code = """import json
import requests
from typing import Dict, List

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def get_data(self, endpoint: str) -> Dict:
        response = self.session.get(f"{self.base_url}/{endpoint}")
        return response.json()

    def post_data(self, endpoint: str, data: Dict) -> Dict:
        response = self.session.post(f"{self.base_url}/{endpoint}", json=data)
        return response.json()

def main():
    client = APIClient("https://api.example.com")
    data = client.get_data("users")
    print(json.dumps(data, indent=2))
"""

    matcher = StreamingFuzzyMatcher(source_code)

    # Simulate finding the get_data method
    query = "def get_data(self, endpoint: str) -> Dict:\n"
    simulate_streaming_input(matcher, query)

    # Show final results
    final_matches = matcher.finish()
    print(f"🏁 Final matches found: {len(final_matches)}")
    for i, match in enumerate(final_matches):
        print(f"   Match {i + 1}: Range({match.start}, {match.end})")


def demonstrate_fuzzy_matching() -> None:
    """Demo fuzzy matching with typos and variations."""
    print("\n\n🔧 FUZZY MATCHING DEMO")
    print("=" * 60)

    source_code = """def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n-1)

def calculate_prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
"""

    matcher = StreamingFuzzyMatcher(source_code)

    # Query with typos:
    query = "def calcuate_fibonaci(n):\n"
    simulate_streaming_input(matcher, query)


def demonstrate_line_hints() -> None:
    """Demo line hint usage for disambiguation."""
    print("\n\n📍 LINE HINTS DEMO")
    print("=" * 60)

    source_code = """# Line 1
def helper_function():
    return "first helper"

# Line 5
def another_helper():
    return "second helper"

# Line 9
def helper_method():
    return "third helper"

# Line 13
def final_helper():
    return "fourth helper"
"""

    # Without line hint
    print("🔍 Without line hint:")
    matcher1 = StreamingFuzzyMatcher(source_code)
    result1 = matcher1.push("def helper")
    if result1:
        matched_text = source_code[result1.start : result1.end].split("\n")[0]
        print(f"✅ Found: {matched_text}")
    else:
        print("❌ No match")

    print()

    # With line hint pointing to line 9
    print("🎯 With line hint (line 9):")
    matcher2 = StreamingFuzzyMatcher(source_code)
    result2 = matcher2.push("def helper", line_hint=9)
    if result2:
        matched_text = source_code[result2.start : result2.end].split("\n")[0]
        print(f"✅ Found: {matched_text}")
    else:
        print("❌ No match")


def demonstrate_multiline_matching() -> None:
    """Demo matching across multiple lines."""
    print("\n\n📄 MULTILINE MATCHING DEMO")
    print("=" * 60)

    source_code = """import os
import sys
import argparse
from pathlib import Path

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    print(f"Processing {input_path} -> {output_path}")
"""

    matcher = StreamingFuzzyMatcher(source_code)

    # Multi-line query
    query_parts = [
        "def main():\n",
        "    setup_logging()\n",
        "    args = parse_arguments()\n",
    ]

    for part in query_parts:
        print(f"📨 Adding: {part!r}")
        result = matcher.push(part)
        if result:
            preview = source_code[result.start : result.start + 50].replace("\n", "\\n")
            print(f"✅ Current best match starts with: {preview!r}...")
        print()


def demonstrate_real_time_editing() -> None:
    """Demo real-time editing scenario."""
    print("\n\n⚡ REAL-TIME EDITING DEMO")
    print("=" * 60)

    source_code = """class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None

    def connect(self):
        # Connect to database
        pass

    def disconnect(self):
        # Disconnect from database
        pass

    def execute_query(self, query, parameters=None):
        # Execute SQL query
        if not self.connection:
            self.connect()
        # Implementation here
        pass

    def fetch_one(self, query, parameters=None):
        # Fetch single result
        result = self.execute_query(query, parameters)
        return result
"""

    print("📝 User starts typing to find execute_query method...")
    matcher = StreamingFuzzyMatcher(source_code)

    # Simulate real-time typing
    typing_sequence = [
        "def exe",
        "cute_",
        "query(",
        "self, ",
        "query, ",
        "parameters=None",
        "):\n",
    ]

    for _i, chunk in enumerate(typing_sequence):
        print(f"⌨️  Typing: {chunk!r}")
        result = matcher.push(chunk)

        if result:
            # Show the matched function signature
            lines = source_code[result.start : result.end].split("\n")
            signature = next((line.strip() for line in lines if "def " in line), "")
            print(f"🎯 Live match: {signature}")
        else:
            print("⏳ Still typing...")

        time.sleep(0.3)  # Simulate typing delay

    print("\n🏁 Final result:")
    final_matches = matcher.finish()
    if final_matches:
        for match in final_matches:
            matched_lines = source_code[match.start : match.end].split("\n")[:3]
            print("✅ Found function:")
            for line in matched_lines:
                if line.strip():
                    print(f"   {line}")


def main() -> None:
    """Run all demonstration examples."""
    print("🚀 STREAMING FUZZY MATCHER EXAMPLES")
    print("🚀 " + "=" * 58)

    demonstrate_exact_matching()
    demonstrate_fuzzy_matching()
    demonstrate_line_hints()
    demonstrate_multiline_matching()
    demonstrate_real_time_editing()

    print("\n\n✨ All demos completed!")
    print("\n💡 Key takeaways:")
    print("   • Matcher works incrementally with streaming input")
    print("   • Handles typos and variations through fuzzy matching")
    print("   • Line hints help disambiguate multiple matches")
    print("   • Suitable for real-time code editing scenarios")
    print("   • Processes incomplete lines until newlines received")


if __name__ == "__main__":
    main()
