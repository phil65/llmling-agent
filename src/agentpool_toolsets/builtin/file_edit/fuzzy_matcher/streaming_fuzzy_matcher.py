"""A streaming fuzzy matcher.

It that can process text chunks incrementally
and return the best match found so far at each step.

This is a Python port of Zed's streaming fuzzy matcher, designed for
real-time code editing and location resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


REPLACEMENT_COST = 1
INSERTION_COST = 3
DELETION_COST = 10

LINE_HINT_TOLERANCE = 200
THRESHOLD = 0.8
SearchDirection = Literal["up", "left", "diagonal"]


@dataclass(frozen=True, slots=True)
class Range:
    """Represents a text range with start and end byte offsets."""

    start: int
    end: int


@dataclass(slots=True)
class SearchState:
    """State for a cell in the dynamic programming matrix."""

    cost: int
    direction: SearchDirection

    def __post_init__(self) -> None:
        self.cost = max(self.cost, 0)


@dataclass
class SearchMatrix:
    """Dynamic programming matrix for edit distance computation."""

    cols: int
    rows: int = 0
    data: list[SearchState] = field(default_factory=list)

    def resize_rows(self, new_rows: int) -> None:
        """Resize matrix to accommodate more query lines."""
        if new_rows <= self.rows:
            return

        old_size = self.rows * self.cols
        new_size = new_rows * self.cols
        self.data.extend([SearchState(0, "diagonal") for _ in range(new_size - old_size)])
        self.rows = new_rows

    def get(self, row: int, col: int) -> SearchState:
        """Get state at matrix position."""
        if row >= self.rows or col >= self.cols or row < 0 or col < 0:
            return SearchState(999999, "diagonal")
        return self.data[row * self.cols + col]

    def set(self, row: int, col: int, state: SearchState) -> None:
        """Set state at matrix position."""
        if row < self.rows and col < self.cols and row >= 0 and col >= 0:
            self.data[row * self.cols + col] = state


class StreamingFuzzyMatcher:
    """A streaming fuzzy matcher that processes text chunks incrementally.

    This matcher accumulates text chunks and performs fuzzy matching against
    a source buffer, returning the best matches found so far.
    """

    def __init__(self, source_text: str) -> None:
        """Initialize matcher with source text to search against.

        Args:
            source_text: The text buffer to search within
        """
        self.source_lines: list[str] = source_text.splitlines()
        self.query_lines: list[str] = []
        self.line_hint: int | None = None
        self.incomplete_line: str = ""
        self.matches: list[Range] = []

        # Initialize matrix with buffer line count + 1
        buffer_line_count = len(self.source_lines)
        self.matrix: SearchMatrix = SearchMatrix(buffer_line_count + 1)

    @property
    def query_lines_list(self) -> list[str]:
        """Returns the accumulated query lines."""
        return self.query_lines.copy()

    def push(self, chunk: str, line_hint: int | None = None) -> Range | None:
        """Push a new chunk of text and get the best match found so far.

        This method accumulates text chunks and processes complete lines.
        Partial lines are buffered internally until a newline is received.

        Args:
            chunk: Text chunk to add to the query
            line_hint: Optional line number hint for match selection

        Returns:
            Range if a match has been found, None otherwise
        """
        if line_hint is not None:
            self.line_hint = line_hint

        # Add chunk to incomplete line buffer
        self.incomplete_line += chunk

        # Process complete lines (everything up to the last newline)
        if "\n" in self.incomplete_line:
            # Find last newline position
            last_newline = self.incomplete_line.rfind("\n")
            complete_part = self.incomplete_line[: last_newline + 1]

            # Split into lines and add to query_lines
            new_lines = complete_part.splitlines()
            self.query_lines.extend(new_lines)

            # Keep remaining incomplete part
            self.incomplete_line = self.incomplete_line[last_newline + 1 :]

            # Update matches with new query lines
            self.matches = self._resolve_location_fuzzy()

        # Return best match
        best_match = self.select_best_match()
        return best_match or (self.matches[0] if self.matches else None)

    def finish(self) -> list[Range]:
        """Finish processing and return all final matches.

        This processes any remaining incomplete line before returning
        the final match results.

        Returns:
            List of all found matches
        """
        # Process any remaining incomplete line
        if self.incomplete_line.strip():
            self.query_lines.append(self.incomplete_line)
            self.incomplete_line = ""
            self.matches = self._resolve_location_fuzzy()

        return self.matches.copy()

    def select_best_match(self) -> Range | None:
        """Return the best match considering line hints.

        Returns:
            Best match range, or None if no suitable match found
        """
        if not self.matches:
            return None

        if len(self.matches) == 1:
            return self.matches[0]

        if self.line_hint is None:
            # Multiple ambiguous matches without hint
            return None

        best_match = None
        best_distance = float("inf")

        for match_range in self.matches:
            # Convert byte offset to approximate line number
            start_line = self._offset_to_line(match_range.start)
            distance = abs(start_line - self.line_hint)

            if distance <= LINE_HINT_TOLERANCE and distance < best_distance:
                best_distance = distance
                best_match = match_range

        return best_match

    def _resolve_location_fuzzy(self) -> list[Range]:
        """Perform fuzzy matching using dynamic programming.

        Returns:
            List of match ranges found in the source text
        """
        new_query_line_count = len(self.query_lines)
        old_query_line_count = max(0, self.matrix.rows - 1)

        if new_query_line_count == old_query_line_count:
            return []

        self.matrix.resize_rows(new_query_line_count + 1)

        # Process only the new query lines
        for row in range(old_query_line_count, new_query_line_count):
            query_line = self.query_lines[row].strip()
            leading_deletion_cost = (row + 1) * DELETION_COST

            # Initialize first column
            self.matrix.set(row + 1, 0, SearchState(leading_deletion_cost, "up"))

            # Process each source line
            for col, source_line in enumerate(self.source_lines):
                line = source_line.strip()

                # Calculate costs for each direction
                up_cost = self.matrix.get(row, col + 1).cost + DELETION_COST
                left_cost = self.matrix.get(row + 1, col).cost + INSERTION_COST

                # Diagonal cost depends on character match
                diagonal_cost = self.matrix.get(row, col).cost
                if not _fuzzy_eq(query_line, line):
                    diagonal_cost += REPLACEMENT_COST

                # Choose minimum cost direction
                if diagonal_cost <= up_cost and diagonal_cost <= left_cost:
                    best_state = SearchState(diagonal_cost, "diagonal")
                elif up_cost <= left_cost:
                    best_state = SearchState(up_cost, "up")
                else:
                    best_state = SearchState(left_cost, "left")

                self.matrix.set(row + 1, col + 1, best_state)

        # Extract matches by backtracking through matrix
        return self._extract_matches()

    def _extract_matches(self) -> list[Range]:
        """Extract match ranges by backtracking through the DP matrix."""
        if not self.query_lines:
            return []

        query_len = len(self.query_lines)

        # Find all reasonable endpoints and score them
        min_cost = float("inf")

        for col in range(len(self.source_lines) + 1):
            state = self.matrix.get(query_len, col)
            min_cost = min(min_cost, state.cost)

        # Accept costs within reasonable range of minimum
        max_acceptable_cost = min(min_cost * 2, query_len * REPLACEMENT_COST * 4)

        candidates: list[tuple[float, Range]] = []
        for col in range(len(self.source_lines) + 1):
            state = self.matrix.get(query_len, col)
            if state.cost <= max_acceptable_cost:
                match_range = self._backtrack_match(query_len, col)
                if match_range:
                    # Score based on cost and match quality
                    quality_score = self._calculate_match_quality(match_range)
                    candidates.append((
                        float(state.cost) - quality_score * 10,
                        match_range,
                    ))

        # Sort by score and return best matches
        candidates.sort(key=lambda x: x[0])
        return [match for _, match in candidates[:5]]  # Limit to top 5 matches

    def _backtrack_match(self, end_row: int, end_col: int) -> Range | None:
        """Backtrack through matrix to find match boundaries."""
        if end_row == 0 or end_col == 0:
            return None

        # Track which source lines were matched
        matched_lines: set[int] = set()
        row, col = end_row, end_col

        # Backtrack to collect matched source lines
        while row > 0 and col > 0:
            state = self.matrix.get(row, col)
            if state.direction == "diagonal":
                # represents a match between query line (row-1) and source line (col-1)
                matched_lines.add(col - 1)
                row -= 1
                col -= 1
            elif state.direction == "up":
                row -= 1
            else:  # LEFT
                col -= 1

        if not matched_lines:
            return None

        # Find contiguous range of matched lines
        matched_list = sorted(matched_lines)
        start_line: int = matched_list[0]
        end_line: int = matched_list[-1]

        # Extend range to include some context for better matches
        context_lines = min(2, len(self.query_lines))
        start_line = max(0, start_line - context_lines // 2)
        end_line = min(len(self.source_lines) - 1, end_line + context_lines // 2)

        start_offset = self._line_to_offset(start_line)
        end_offset = self._line_to_offset(end_line + 1)

        return Range(start_offset, end_offset)

    def _calculate_match_quality(self, match_range: Range) -> float:
        """Calculate quality score for a match (higher is better)."""
        if not self.query_lines:
            return 0.0

        source_text = "\n".join(self.source_lines)
        matched_text = source_text[match_range.start : match_range.end]
        matched_lines = matched_text.split("\n")

        score = 0.0
        query_words: set[str] = set()
        for line in self.query_lines:
            query_words.update(line.strip().split())

        # Score based on word overlap
        for line in matched_lines:
            line_words = set(line.strip().split())
            overlap = len(query_words & line_words)
            if len(query_words) > 0:
                score += overlap / len(query_words)

        # Bonus for exact line matches
        for query_line in self.query_lines:
            normalized_query = " ".join(query_line.strip().split())
            for matched_line in matched_lines:
                normalized_matched = " ".join(matched_line.strip().split())
                if normalized_query == normalized_matched:
                    score += 2.0

        return score

    def _line_to_offset(self, line_num: int) -> int:
        """Convert line number to byte offset."""
        if line_num <= 0:
            return 0

        # Calculate offset by summing lengths of previous lines
        offset = 0
        for i in range(min(line_num, len(self.source_lines))):
            offset += len(self.source_lines[i]) + 1  # +1 for newline

        if line_num >= len(self.source_lines):
            return offset

        return max(0, offset - 1) if offset > 0 else 0

    def _offset_to_line(self, offset: int) -> int:
        """Convert byte offset to approximate line number."""
        if offset <= 0:
            return 0

        current_offset = 0
        for i, line in enumerate(self.source_lines):
            if current_offset + len(line) + 1 > offset:
                return i
            current_offset += len(line) + 1

        return len(self.source_lines) - 1


def _fuzzy_eq(left: str, right: str) -> bool:
    """Check if two strings are fuzzy equal using normalized Levenshtein distance.

    Args:
        left: First string to compare
        right: Second string to compare

    Returns:
        True if strings are similar enough (>= 0.8 similarity)
    """
    if not left and not right:
        return True
    if not left or not right:
        return False

    # Normalize whitespace for better matching
    left_normalized = " ".join(left.split())
    right_normalized = " ".join(right.split())

    # Quick exact match after normalization
    if left_normalized == right_normalized:
        return True

    # Quick check: if length difference is too large, they can't be similar enough
    max_len = max(len(left_normalized), len(right_normalized))
    if max_len == 0:
        return True

    min_levenshtein = abs(len(left_normalized) - len(right_normalized))
    min_normalized_similarity = 1.0 - (min_levenshtein / max_len)

    if min_normalized_similarity < THRESHOLD:
        return False

    # Calculate actual Levenshtein distance
    distance = _levenshtein_distance(left_normalized, right_normalized)
    normalized_similarity = 1.0 - (distance / max_len)

    return normalized_similarity >= THRESHOLD


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance between the strings
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Create matrix
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]
