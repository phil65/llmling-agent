"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

from importlib.util import find_spec
import os
from typing import TYPE_CHECKING, Any

from toprompt import to_prompt
from upathtools import read_path, to_upath

from llmling_agent.models.content import (
    BaseContent,
    BaseImageContent,
    BasePDFContent,
    ImageBase64Content,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.models.content import (
        Content,
    )


def is_pil_image(obj: Any) -> bool:
    """Check if object is a PIL.Image.Image instance without direct import."""
    if not find_spec("PIL"):
        return False
    import PIL.Image

    return isinstance(obj, PIL.Image.Image)


async def convert_prompts(
    prompts: Sequence[AnyPromptType | PIL.Image.Image | os.PathLike[str] | Content],
) -> list[str | Content]:
    """Convert prompts to our internal format.

    Handles:
    - PIL Images -> ImageBase64Content
    - UPath/PathLike -> Auto-detect and convert to appropriate Content
    - Regular prompts -> str via to_prompt
    - Content objects -> pass through
    """
    result: list[str | Content] = []
    for p in prompts:
        match p:
            case _ if is_pil_image(p):
                # Only convert PIL images if PIL is available
                result.append(ImageBase64Content.from_pil_image(p))  # type: ignore

            case os.PathLike():
                from mimetypes import guess_type

                path_obj = to_upath(p)
                mime_type, _ = guess_type(str(path_obj))

                match mime_type:
                    case "application/pdf":
                        content: Content = await BasePDFContent.from_path(path_obj)
                        result.append(content)
                    case str() if mime_type.startswith("image/"):
                        content = await BaseImageContent.from_path(path_obj)
                        result.append(content)
                    case _:
                        # Non-media or unknown type
                        text = await read_path(path_obj)
                        result.append(text)

            case _ if not isinstance(p, BaseContent):
                result.append(await to_prompt(p))
            case _:
                result.append(p)  # type: ignore
    return result


async def format_prompts(prompts: Sequence[str | Content]) -> str:
    """Format prompts for human readability using to_prompt."""
    from toprompt import to_prompt

    parts = [await to_prompt(p) for p in prompts]
    return "\n\n".join(parts)
