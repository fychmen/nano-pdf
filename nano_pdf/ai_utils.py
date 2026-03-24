import os
import base64
import re
from io import BytesIO
from typing import List
from PIL import Image
import httpx

MODEL = os.getenv("NANO_PDF_MODEL", "google/gemini-3-pro-image-preview")


def _get_proxy_url():
    """Get OpenRouter proxy URL from environment (set by OpenClaw sandbox)."""
    url = os.getenv("OPENROUTER_PROXY_URL")
    if not url or "/none" in url:
        raise ValueError(
            "OPENROUTER_PROXY_URL not configured. "
            "nano-pdf requires per-client OpenRouter proxy for image editing."
        )
    return url


def _image_to_base64_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 data URL."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _call_openrouter(messages: list, proxy_url: str) -> dict:
    """Call OpenRouter API through caddy proxy (key injected by proxy)."""
    api_url = f"{proxy_url}/api/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "modalities": ["text", "image"],
    }
    resp = httpx.post(
        api_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_image_from_response(data: dict) -> Image.Image:
    """Extract generated image from OpenRouter response."""
    msg = data.get("choices", [{}])[0].get("message", {})

    # Check message.images field (OpenRouter puts images here)
    images = msg.get("images", [])
    if images:
        for img in images:
            if img.get("type") == "image_url":
                url = img.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    b64 = url.split(",", 1)[1]
                    return Image.open(BytesIO(base64.b64decode(b64)))

    # Check content as multimodal array
    content = msg.get("content", "")
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    b64 = url.split(",", 1)[1]
                    return Image.open(BytesIO(base64.b64decode(b64)))

    # Check content as string with base64
    if isinstance(content, str):
        match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
        if match:
            return Image.open(BytesIO(base64.b64decode(match.group(1))))

    raise RuntimeError("No image in response from model.")


def generate_edited_slide(
    target_image: Image.Image,
    style_reference_images: List[Image.Image],
    full_text_context: str,
    user_prompt: str,
    resolution: str = "4K",
    enable_search: bool = False,
) -> Image.Image:
    """Edit a slide image using Gemini via OpenRouter."""
    proxy_url = _get_proxy_url()

    content_parts = []

    # Target image
    content_parts.append({
        "type": "image_url",
        "image_url": {"url": _image_to_base64_url(target_image)},
    })

    # Style references
    if style_reference_images:
        for ref_img in style_reference_images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": _image_to_base64_url(ref_img)},
            })

    # Text prompt with context
    prompt_text = user_prompt
    if style_reference_images:
        prompt_text += "\n\nMatch the visual style (fonts, colors, layout) of the reference images."
    if full_text_context:
        prompt_text += f"\n\nDOCUMENT CONTEXT:\n{full_text_context}"
    prompt_text += f"\n\nOutput resolution: {resolution}."

    content_parts.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content_parts}]

    try:
        data = _call_openrouter(messages, proxy_url)
    except httpx.HTTPStatusError as e:
        error_text = e.response.text.lower()
        if "quota" in error_text or "billing" in error_text:
            raise RuntimeError(
                "API Error: Usage limit reached. "
                "Please check your plan's usage limits."
            ) from e
        raise RuntimeError(f"API Error: {e.response.status_code} {e.response.text}") from e

    return _extract_image_from_response(data)


def generate_new_slide(
    style_reference_images: List[Image.Image],
    user_prompt: str,
    full_text_context: str = "",
    resolution: str = "4K",
    enable_search: bool = False,
) -> Image.Image:
    """Generate a new slide using Gemini via OpenRouter."""
    proxy_url = _get_proxy_url()

    content_parts = []

    if style_reference_images:
        for ref_img in style_reference_images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": _image_to_base64_url(ref_img)},
            })

    prompt_text = user_prompt
    if style_reference_images:
        prompt_text += "\n\nMatch the visual style (fonts, colors, layout) of the reference images."
    if full_text_context:
        prompt_text += f"\n\nDOCUMENT CONTEXT:\n{full_text_context}"
    prompt_text += f"\n\nOutput resolution: {resolution}."

    content_parts.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content_parts}]

    try:
        data = _call_openrouter(messages, proxy_url)
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API Error: {e.response.status_code} {e.response.text}") from e

    return _extract_image_from_response(data)
