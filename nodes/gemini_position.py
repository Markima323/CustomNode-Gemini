"""
ComfyUI-JM-Gemini-API Text Prompt Processor Node
Take an input text, apply a prompt-driven transformation using Gemini, and return the processed text.
"""

import logging
from google.genai import types
from .utils import create_ai_studio_client

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "gemini-3-flash-preview"


class JMGeminiPromptTextProcessor:
    """
    ComfyUI custom node for processing text with a prompt using Gemini.
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
                    "placeholder": "Describe how the input text should be handled"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste the text to process"
                }),
                "model": (["gemini-3-flash-preview"], {
                    "default": DEFAULT_TEXT_MODEL
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_text"
    CATEGORY = "JM-Gemini"

    def process_text(self, gemini_api_key, prompt, text, model=DEFAULT_TEXT_MODEL, seed=0):
        """
        Call Gemini with a prompt and input text, returning the processed text.
        """
        if not gemini_api_key or not gemini_api_key.strip():
            raise ValueError("Gemini API key is required")

        prompt_value = prompt.strip() if prompt else ""
        if not prompt_value:
            raise ValueError("Prompt is required to describe how to process the text")

        input_text = text.strip() if text else ""
        if not input_text:
            raise ValueError("Input text is required")

        try:
            client = create_ai_studio_client(gemini_api_key)

            contents = [
                prompt_value,
                f"Input text:\n{input_text}"
            ]

            config = types.GenerateContentConfig(
                response_modalities=["TEXT"]
            )

            logger.info(f"[JM-Gemini] Processing text with model={model}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            processed_text = self._extract_text(response)
            return (processed_text,)

        except Exception as e:
            logger.exception(f"[JM-Gemini] Failed to process text: {e}")
            raise RuntimeError(f"Failed to process text: {str(e)}")

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
                if content:
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        if getattr(part, "text", None):
                            texts.append(part.text)

        if hasattr(response, "parts"):
            for part in (response.parts or []):
                if getattr(part, "text", None):
                    texts.append(part.text)

        merged = "\n\n".join(t.strip() for t in texts if t and t.strip())
        if merged:
            return merged

        raise RuntimeError("No text content received from Gemini API")


NODE_CLASS_MAPPINGS = {
    "JMGeminiPromptTextProcessor": JMGeminiPromptTextProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMGeminiPromptTextProcessor": "JM Gemini Prompt Text Processor"
}
