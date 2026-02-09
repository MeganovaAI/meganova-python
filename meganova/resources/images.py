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
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 4,
        guidance_scale: float = 3.5,
        seed: int = -1,
    ) -> ImageGenerationResponse:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Image generation model name.
            width: Image width in pixels.
            height: Image height in pixels.
            num_steps: Number of diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed (-1 for random).

        Returns:
            ImageGenerationResponse containing base64-encoded image(s).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": width,
            "height": height,
        }
        data = self._transport.request("POST", "/images/generation", json=payload)
        return ImageGenerationResponse(**data)
