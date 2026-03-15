"""Image description tool using a vision model."""

import base64
from pathlib import Path
from typing import Any

import httpx

from nanobot.agent.tools import Tool

VISION_API_BASE = "http://127.0.0.1:11435/v1"
VISION_MODEL = "llama3.2-vision"


class DescribeImageTool(Tool):
    name = "describe_image"
    description = (
        "Describe the contents of an image file using a vision model. "
        "Returns a detailed text description of the image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the image file (jpg, png, gif, webp, etc.)"},
            "question": {"type": "string", "description": "Optional question to ask about the image"},
        },
        "required": ["path"],
    }

    async def execute(self, path: str, question: str = "Describe this image in detail.", **kw: Any) -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: Image file not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"

        # Read and base64 encode the image
        try:
            image_data = p.read_bytes()
            b64 = base64.standard_b64encode(image_data).decode("utf-8")
        except Exception as e:
            return f"Error reading image: {e}"

        # Detect MIME type from extension
        suffix = p.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
        }
        mime = mime_map.get(suffix, "image/jpeg")

        # Call vision model
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": 1024,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{VISION_API_BASE}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            return "Error: Vision model service not available (port 11435). Start it with: OLLAMA_MODELS=/bigtemp/fzv6en/ollama_models OLLAMA_HOST=127.0.0.1:11435 ollama serve &"
        except Exception as e:
            return f"Error calling vision model: {e}"
