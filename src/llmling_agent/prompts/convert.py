"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic_ai import AudioUrl, BinaryContent, BinaryImage, DocumentUrl, ImageUrl, VideoUrl
from toprompt import to_prompt
from upathtools import UPath, read_path, to_upath


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import UserContent

    from llmling_agent.common_types import PromptCompatible


async def convert_prompts(
    prompts: Sequence[PromptCompatible],
) -> list[UserContent]:
    """Convert prompts to pydantic-ai UserContent format.

    Handles:
    - PIL Images -> BinaryImage
    - UPath/PathLike -> Auto-detect and convert to appropriate Content
    - Regular prompts -> str via to_prompt
    - pydantic-ai content objects -> pass through
    """
    result: list[UserContent] = []
    for p in prompts:
        match p:
            case os.PathLike() | UPath():
                from mimetypes import guess_type

                path_obj = to_upath(p)
                mime_type, _ = guess_type(str(path_obj))

                match mime_type:
                    case "application/pdf":
                        # For http(s) URLs, use DocumentUrl; otherwise read binary
                        if path_obj.protocol in {"http", "https"}:
                            result.append(DocumentUrl(url=str(path_obj)))
                        else:
                            data = await read_path(path_obj, mode="rb")
                            result.append(BinaryContent(data=data, media_type="application/pdf"))
                    case str() if mime_type.startswith("image/"):
                        if path_obj.protocol in {"http", "https"}:
                            result.append(ImageUrl(url=str(path_obj)))
                        else:
                            data = await read_path(path_obj, mode="rb")
                            result.append(BinaryImage(data=data, media_type=mime_type))
                    case str() if mime_type.startswith("audio/"):
                        if path_obj.protocol in {"http", "https"}:
                            result.append(AudioUrl(url=str(path_obj)))
                        else:
                            data = await read_path(path_obj, mode="rb")
                            result.append(BinaryContent(data=data, media_type=mime_type))
                    case str() if mime_type.startswith("video/"):
                        if path_obj.protocol in {"http", "https"}:
                            result.append(VideoUrl(url=str(path_obj)))
                        else:
                            # Video as binary content
                            data = await read_path(path_obj, mode="rb")
                            result.append(BinaryContent(data=data, media_type=mime_type))
                    case _:
                        # Non-media or unknown type - read as text
                        text = await read_path(path_obj)
                        result.append(text)

            case (
                str()
                | ImageUrl()
                | AudioUrl()
                | DocumentUrl()
                | VideoUrl()
                | BinaryContent()
                | BinaryImage()
            ):
                # Already a valid UserContent type
                result.append(p)

            case _:
                # Use to_prompt for anything else (PIL images, pydantic models, etc.)
                result.append(await to_prompt(p))
    return result


async def format_prompts(prompts: Sequence[UserContent]) -> str:
    """Format prompts for human readability using to_prompt."""
    from toprompt import to_prompt

    parts = [await to_prompt(p) for p in prompts]
    return "\n\n".join(parts)
