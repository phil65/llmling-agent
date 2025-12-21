"""Skills configuration."""

from dataclasses import dataclass


@dataclass
class Skill:
    """Skill configuration."""

    url: str
    name: str


dev_browser = Skill(
    url="https://github.com/SawyerHood/dev-browser/tree/main/skills/dev-browser",
    name="dev-browser",
)
