"""Image processing utilities for the fsspec toolset."""

from __future__ import annotations


def resize_image_if_needed(
    data: bytes,
    media_type: str,
    max_size: int,
    jpeg_quality: int = 85,
) -> tuple[bytes, str, str | None]:
    """Resize image if it exceeds max_size, preserving aspect ratio.

    Args:
        data: Raw image bytes
        media_type: MIME type of the image
        max_size: Maximum width/height in pixels
        jpeg_quality: Quality for JPEG output (1-100)

    Returns:
        Tuple of (image_data, media_type, dimension_note).
        dimension_note is None if no resizing was needed, otherwise contains
        a message about the resize for the model to map coordinates.
    """
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(data))
    orig_w, orig_h = img.size

    if orig_w <= max_size and orig_h <= max_size:
        return data, media_type, None

    # Calculate new size maintaining aspect ratio
    ratio = min(max_size / orig_w, max_size / orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Determine output format - preserve original if possible
    out = BytesIO()
    if media_type in ("image/jpeg", "image/jpg"):
        resized.save(out, format="JPEG", quality=jpeg_quality)
        out_type = "image/jpeg"
    elif media_type == "image/png":
        resized.save(out, format="PNG")
        out_type = "image/png"
    elif media_type == "image/webp":
        resized.save(out, format="WEBP", quality=jpeg_quality)
        out_type = "image/webp"
    elif media_type == "image/gif":
        # GIF resize loses animation, convert to PNG
        resized.save(out, format="PNG")
        out_type = "image/png"
    else:
        # Default to JPEG for unknown formats
        resized.save(out, format="JPEG", quality=jpeg_quality)
        out_type = "image/jpeg"

    scale = orig_w / new_w
    note = f"[Image resized: {orig_w}x{orig_h} → {new_w}x{new_h}, scale={scale:.2f}x]"
    return out.getvalue(), out_type, note
