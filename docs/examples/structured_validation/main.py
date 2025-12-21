# /// script
# dependencies = ["agentpool"]
# ///

"""Example showing structured agent output validation using movie reviews."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from schemez import Schema

from agentpool import Agent
from agentpool.docs.utils import run


class MovieReview(Schema):
    """Basic movie review."""

    score: float  # Any score from 1-10
    excitement: float  # Reviewer's excitement level
    would_recommend: bool  # Basic recommendation
    key_points: list[str]  # Main points about the movie


class RavingReview(MovieReview):
    """Only truly enthusiastic reviews.

    A review that shows genuine excitement about the movie:
    - High score (at least 8.5)
    - High excitement level (at least 0.9)
    - Must recommend
    - At least 3 positive points
    """

    score: float = Field(ge=8.5)  # Must really like it
    excitement: float = Field(ge=0.9)  # Must be super excited
    would_recommend: Literal[True] = True  # Must recommend
    key_points: list[str] = Field(min_length=3)  # At least 3 points


async def main() -> None:
    # Create movie critic agent
    critic = Agent(
        name="movie_critic",
        model="openai:gpt-5-nano",
        output_type=MovieReview,
        system_prompt=(
            "You are an enthusiastic but honest movie critic. "
            "You love great movies but won't pretend to like bad ones."
        ),
    )

    # Test with different movies
    movies = [
        "The Room by Tommy Wiseau",
        "The Lord of the Rings: The Fellowship of the Ring",
    ]

    for movie in movies:
        print(f"\n{'=' * 50}\n🎬 Reviewing: {movie}\n{'=' * 50}\n")

        # Get basic review
        result = await critic.run(f"What did you think of {movie}?")
        review = result.content

        print(f"\n📊 Score: {review.score}/10")
        print(f"🎯 Excitement Level: {review.excitement}")
        print(f"👍 Recommend: {'Yes' if review.would_recommend else 'No'}")
        print("\n🔑 Key Points:")
        for point in review.key_points:
            print(f"  • {point}")

        # Check if it's a truly raving review
        is_raving = await critic.validate_against(f"What did you think of {movie}?", RavingReview)
        print(f"\n🌟 Absolute Rave Review: {'YES!' if is_raving else 'Nope.'}")


if __name__ == "__main__":
    run(main())


"""
Example output:

Reviewing: The Room by Tommy Wiseau
Basic review: MovieReview(
    score=2.5,
    excitement=0.3,
    would_recommend=False,
    key_points=['So bad it\'s good', 'Unintentionally hilarious', 'Cult classic']
)
Is absolutely raving about it: False

Reviewing: The Lord of the Rings: The Fellowship of the Ring
Basic review: MovieReview(
    score=9.5,
    excitement=0.95,
    would_recommend=True,
    key_points=[
        'Masterful adaptation of Tolkien\'s work',
        'Groundbreaking special effects',
        'Exceptional performances',
        'Beautiful score by Howard Shore'
    ]
)
Is absolutely raving about it: True
"""
