"""
ComfyUI-JM-Gemini-API Vision-to-Text Node
Use Gemini 3 Flash Preview to read an image and return text guided by a prompt.
"""

import logging
from google.genai import types
from .utils import create_ai_studio_client, tensor2pil

logger = logging.getLogger(__name__)

GEMINI_FLASH_PREVIEW_MODEL = "gemini-3-flash-preview"


class JMGeminiFlashPreviewVisionToText:
    """
    ComfyUI custom node for extracting text from an image using Gemini 3 Flash Preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Gemini API key",
                    "password": True
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe what you want the model to answer about the image"
                }),
                "image": ("IMAGE",),
            },
            "optional": {
                # seed is only used by ComfyUI to force re-execution; it is not sent to Gemini
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "JM_Gemini"

    def generate_text(self, gemini_api_key, prompt, image, seed=0):
        """
        Call Gemini 3 Flash Preview with an image and prompt to get a text response.
        """
        if not gemini_api_key or not gemini_api_key.strip():
            raise ValueError("Gemini API key is required")

        if image is None:
            raise ValueError("An input image is required")

        prompt_value = prompt.strip() if prompt else ""
        if not prompt_value:
            raise ValueError("Prompt is required to guide the model output")

        try:
            client = create_ai_studio_client(gemini_api_key)

            pil_image = self._tensor_to_single_pil(image)
            contents = [prompt_value, pil_image]

            config = types.GenerateContentConfig(
                response_modalities=["TEXT"]
            )

            logger.info("[JM-Gemini] Calling Gemini 3 Flash Preview for vision-to-text")
            response = client.models.generate_content(
                model=GEMINI_FLASH_PREVIEW_MODEL,
                contents=contents,
                config=config,
            )

            text = self._extract_text(response)
            return (text,)

        except Exception as e:
            logger.exception(f"[JM-Gemini] Failed to generate text: {e}")
            raise RuntimeError(f"Failed to generate text: {str(e)}")

    def _tensor_to_single_pil(self, image_tensor):
        """
        Convert a ComfyUI IMAGE tensor to a single PIL image (use the first frame if batched).
        """
        try:
            if hasattr(image_tensor, "shape") and len(image_tensor.shape) == 4 and image_tensor.shape[0] > 1:
                # Keep only the first image in the batch for analysis
                image_tensor = image_tensor[0:1]
            return tensor2pil(image_tensor)
        except Exception as e:
            logger.error(f"[JM-Gemini] Could not process input image: {e}")
            raise ValueError(f"Failed to process input image: {e}")

    def _extract_text(self, response):
        """
        Extract text content from Gemini response.
        """
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        texts = []

        if hasattr(response, "candidates"):
            for candidate in response.candidates or []:
                content = getattr(candidate, "content", None)
                if content and hasattr(content, "parts"):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            texts.append(part.text)

        if hasattr(response, "parts"):
            for part in response.parts:
                if getattr(part, "text", None):
                    texts.append(part.text)

        merged = "\n\n".join(t.strip() for t in texts if t and t.strip())
        if merged:
            return merged

        raise RuntimeError("No text content received from Gemini API")


NODE_CLASS_MAPPINGS = {
    "JMGeminiFlashPreviewVisionToText": JMGeminiFlashPreviewVisionToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMGeminiFlashPreviewVisionToText": "JM Gemini 3 Flash Vision-to-Text"
}
