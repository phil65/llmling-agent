"""Download tool."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Literal
from urllib.parse import urlparse

from annotated_types import Ge, Le

from llmling_agent import AgentContext  # noqa: TC001


PaperType = Literal[
    "letter", "legal", "tabloid", "ledger", "a0", "a1", "a2", "a3", "a4", "a5", "a6"
]


async def download_file(
    context: AgentContext,
    url: str,
    target_dir: str = "downloads",
    chunk_size: int = 8192,
    verify_ssl: bool = False,  # For testing, in prod should be True
) -> str:
    """Download a file and return status information."""
    import httpx
    from upath import UPath

    start_time = time.time()
    target_path = UPath(target_dir)
    target_path.mkdir(exist_ok=True)

    filename = UPath(urlparse(url).path).name or "downloaded_file"
    full_path = target_path / filename
    try:
        async with (
            httpx.AsyncClient(verify=verify_ssl) as client,
            client.stream("GET", url, timeout=30.0) as response,
        ):
            response.raise_for_status()

            total = (
                int(response.headers["Content-Length"])
                if "Content-Length" in response.headers
                else None
            )

            with full_path.open("wb") as f:
                size = 0
                async for chunk in response.aiter_bytes(chunk_size):
                    size += len(chunk)
                    f.write(chunk)

                    if total and (size % (chunk_size * 100) == 0 or size == total):
                        progress = size / total * 100
                        speed_mbps = (size / 1_048_576) / (time.time() - start_time)
                        msg = f"\r{filename}: {progress:.1f}% ({speed_mbps:.1f} MB/s)"
                        if context.report_progress:
                            await context.report_progress(progress, 100, msg)
                        await asyncio.sleep(0)

        duration = time.time() - start_time
        size_mb = size / 1_048_576

        return f"Downloaded {filename} ({size_mb:.1f}MB) at {size_mb / duration:.1f} MB/s"

    except httpx.ConnectError as e:
        return f"Connection error downloading {url}: {e}"
    except httpx.TimeoutException:
        return f"Timeout downloading {url}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} downloading {url}"
    except Exception as e:  # noqa: BLE001
        return f"Error downloading {url}: {e!s}"


def mermaid_image(
    mermaid_code: str,
    image_type: Literal["jpeg", "png", "webp", "svg", "pdf"] | None = None,
    pdf_fit: bool = False,
    pdf_landscape: bool = False,
    pdf_paper: PaperType | None = None,
    bg_color: str | None = None,
    theme: Literal["default", "neutral", "dark", "forest"] | None = None,
    width: int | None = None,
    height: int | None = None,
    scale: Annotated[float, Ge(1), Le(3)] | None = None,
) -> bytes:
    """Generate an image of a Mermaid diagram.

    Returns data for a PDF.

    Args:
        mermaid_code: Mermaid code to generate the image for.
        image_type: The image type to generate. If unspecified, default to `'jpeg'`.
        pdf_fit: When using image_type='pdf', whether to fit the diagram to the PDF page.
        pdf_landscape: When using image_type='pdf', whether to use landscape orientation.
            This has no effect if using `pdf_fit`.
        pdf_paper: When using image_type='pdf', the paper size of the PDF.
        bg_color: The background color of the diagram. If None, transparent is used.
            The color value is interpreted as a hexadecimal color code by default,
            but you can also use named colors by prefixing the value with '!'.
            For example, valid choices include `bg_color='!white'` or `bg_color='FF0000'`.
        theme: The theme of the diagram. Defaults to 'default'.
        width: The width of the diagram.
        height: The height of the diagram.
        scale: The scale of the diagram. The scale must be a number between 1 and 3,
        and you can only set a scale if one or both of width and height are set.
    """
    import base64

    import httpx

    code_base64 = base64.b64encode(mermaid_code.encode()).decode()

    params: dict[str, str] = {}
    if image_type == "pdf":
        url = f"https://mermaid.ink/pdf/{code_base64}"
        if pdf_fit:
            params["fit"] = ""
        if pdf_landscape:
            params["landscape"] = ""
        if pdf_paper:
            params["paper"] = pdf_paper
    else:
        url = f"https://mermaid.ink/img/{code_base64}"

        if image_type:
            params["type"] = image_type

    if bg_color:
        params["bgColor"] = bg_color
    if theme:
        params["theme"] = theme
    if width:
        params["width"] = str(width)
    if height:
        params["height"] = str(height)
    if scale:
        params["scale"] = str(scale)

    response = httpx.get(url, params=params)
    response.raise_for_status()
    return response.content
