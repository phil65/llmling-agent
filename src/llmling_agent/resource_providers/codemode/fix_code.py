"""Prepare LLM output to be in proper shape and executable."""

from __future__ import annotations


def fix_code(python_code: str) -> str:
    """Fix code to be executable."""
    # Simplified execution: require main() function pattern
    if "async def main(" not in python_code:
        # Auto-wrap code in main function, ensuring last expression is returned
        lines = python_code.strip().splitlines()
        if lines:
            # Check if last line is an expression (not a statement)
            last_line = lines[-1].strip()
            if last_line and not any(
                last_line.startswith(kw)
                for kw in [
                    "import ",
                    "from ",
                    "def ",
                    "class ",
                    "if ",
                    "for ",
                    "while ",
                    "try ",
                    "with ",
                    "async def ",
                ]
            ):
                # Last line looks like an expression, add return
                lines[-1] = f"    return {last_line}"
                indented_lines = [f"    {line}" for line in lines[:-1]] + [lines[-1]]
            else:
                indented_lines = [f"    {line}" for line in lines]
            python_code = "async def main():\n" + "\n".join(indented_lines)
        else:
            python_code = "async def main():\n    pass"
    return python_code
