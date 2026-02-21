from typing import Optional
from ..transport import SyncTransport
from ..models.images import ImageGenerationResponse


class ImagesResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        n: int = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        response_format: Optional[str] = None,
        style: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt (OpenAI-compatible).

        Args:
            prompt: Text description of the desired image.
            model: Image generation model name.
            n: Number of images to generate.
            size: Image size as "WxH" string (e.g. "1024x1024").
            quality: Quality level ("standard" or "hd").
            response_format: "url" or "b64_json".
            style: Style preset (e.g. "vivid", "natural").
            width: Image width (legacy; prefer size).
            height: Image height (legacy; prefer size).
            num_steps: Number of diffusion steps (provider-specific).
            guidance_scale: Guidance scale (provider-specific).
            seed: Random seed (provider-specific).
            **kwargs: Additional provider-specific parameters.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
        }

        # Resolve size: explicit size takes priority, then width/height
        if size is not None:
            payload["size"] = size
        elif width is not None and height is not None:
            payload["size"] = f"{width}x{height}"

        for key, value in [
            ("quality", quality),
            ("response_format", response_format),
            ("style", style),
            ("num_steps", num_steps),
            ("guidance_scale", guidance_scale),
            ("seed", seed),
        ]:
            if value is not None:
                payload[key] = value

        payload.update(kwargs)

        data = self._transport.request("POST", "/images/generations", json=payload)
        return ImageGenerationResponse(**data)
