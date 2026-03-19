"""
ComfyUI-JM-Gemini-API Mixed Image-and-Text Node
Combine an input image with two text snippets and a prompt, then generate a new image using Gemini 2.5 Flash Image.
"""

import io
import logging
import mimetypes
import os
import time
from PIL import Image
from google import genai
from google.genai import types

from .utils import tensor2pil, pil2tensor, get_output_dir

logger = logging.getLogger(__name__)

GEMINI_FLASH_IMAGE_MODEL = "gemini-2.5-flash-image"

ASPECT_RATIO_RESOLUTIONS = {
    "1:1": "1024x1024",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "3:4": "864x1184",
    "4:3": "1184x864",
    "4:5": "896x1152",
    "5:4": "1152x896",
    "9:16": "768x1344",
    "16:9": "1344x768",
    "21:9": "1536x672"
}


class JMGeminiFlashImageWithTexts:
    """
    ComfyUI custom node that merges an image and two text inputs to guide Gemini 2.5 Flash Image.
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
                    "placeholder": "Overall instructions for the generated image"
                }),
                "text_one": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "First text detail to include"
                }),
                "text_two": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Second text detail to include"
                }),
                "image": ("IMAGE",),
                "aspect_ratio": ([
                    "1:1", "2:3", "3:2", "3:4", "4:3",
                    "4:5", "5:4", "9:16", "16:9", "21:9"
                ], {
                    "default": "1:1"
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "JM-Gemini"

    def generate_image(self, gemini_api_key, prompt, text_one, text_two, image,
                      aspect_ratio="1:1", seed=0):
        """
        Call Gemini 2.5 Flash Image with aggregated text and an input image to create a new image.
        """
        if not gemini_api_key or not gemini_api_key.strip():
            raise ValueError("Gemini API key is required")

        if image is None:
            raise ValueError("An input image is required")

        prompt_value = prompt.strip() if prompt else ""
        if not prompt_value:
            raise ValueError("Prompt is required to describe the desired output")

        text_one_value = text_one.strip() if text_one else ""
        text_two_value = text_two.strip() if text_two else ""

        if not text_one_value and not text_two_value:
            raise ValueError("At least one text input is required to guide the composition")

        try:
            client = genai.Client(api_key=gemini_api_key)
            base_image = self._tensor_to_single_pil(image)

            text_details = []
            if text_one_value:
                text_details.append(f"Text 1:\n{text_one_value}")
            if text_two_value:
                text_details.append(f"Text 2:\n{text_two_value}")

            combined_prompt = prompt_value
            if text_details:
                combined_prompt = f"{prompt_value}\n\nUse these text details:\n" + "\n\n".join(text_details)

            contents = [combined_prompt, base_image]

            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio
                )
            )

            resolution_info = ASPECT_RATIO_RESOLUTIONS.get(aspect_ratio, "1024x1024")
            logger.info(
                "[JM-Gemini] Generating image with model=%s aspect_ratio=%s (~%s)",
                GEMINI_FLASH_IMAGE_MODEL,
                aspect_ratio,
                resolution_info
            )

            response = client.models.generate_content(
                model=GEMINI_FLASH_IMAGE_MODEL,
                contents=contents,
                config=config,
            )

            output_dir = get_output_dir()
            generated_image = self._extract_image(response, output_dir, aspect_ratio)
            return (generated_image,)

        except Exception as e:
            logger.exception("[JM-Gemini] Failed to generate image with texts: %s", e)
            raise RuntimeError(f"Failed to generate image: {str(e)}")

    def _tensor_to_single_pil(self, image_tensor):
        """
        Convert a ComfyUI IMAGE tensor to a single PIL image (uses the first frame if batched).
        """
        if hasattr(image_tensor, "shape") and len(image_tensor.shape) == 4 and image_tensor.shape[0] > 1:
            image_tensor = image_tensor[0:1]
        return tensor2pil(image_tensor)

    def _extract_image(self, response, output_dir, aspect_ratio):
        """
        Extract the first image from the Gemini response and save it to the output directory.
        """
        if not hasattr(response, "parts") or not response.parts:
            raise RuntimeError("No response parts received from Gemini API")

        for idx, part in enumerate(response.parts):
            image = None

            try:
                image = part.as_image()
            except AttributeError:
                image = None
            except Exception as e:
                logger.warning("[JM-Gemini] Could not read part %s as image: %s", idx, e)

            if image is None and hasattr(part, "inline_data"):
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    try:
                        image = Image.open(io.BytesIO(inline_data.data))
                    except Exception as e:
                        logger.warning("[JM-Gemini] Failed to decode inline image data for part %s: %s", idx, e)

            if image is not None:
                image = self._to_pil_image(image)
                if image is None:
                    logger.warning("[JM-Gemini] Unable to convert response part %s to PIL image", idx)
                    continue
                timestamp = int(time.time())
                aspect_slug = aspect_ratio.replace(":", "x")
                extension = ".png"

                try:
                    mime_type = getattr(getattr(part, "inline_data", None), "mime_type", None)
                    if mime_type:
                        extension = mimetypes.guess_extension(mime_type) or extension
                except Exception:
                    pass

                file_name = f"gemini25flash_mix_{aspect_slug}_{timestamp}{extension}"
                file_path = os.path.join(output_dir, file_name)

                try:
                    image.save(file_path)
                    logger.info("[JM-Gemini] Saved generated image to %s", file_path)
                except Exception as e:
                    logger.warning("[JM-Gemini] Failed to save image to disk: %s", e)

                return pil2tensor(image)

        raise RuntimeError("No images were generated. Please adjust your prompt and try again.")

    def _to_pil_image(self, image_obj):
        """
        Ensure the returned image object is a PIL Image.
        """
        if isinstance(image_obj, Image.Image):
            return image_obj

        # google.genai image objects may expose helper accessors
        if hasattr(image_obj, "as_pil_image"):
            try:
                converted = image_obj.as_pil_image()
                if isinstance(converted, Image.Image):
                    return converted
            except Exception as e:
                logger.debug("[JM-Gemini] as_pil_image conversion failed: %s", e)

        for attr in ("_pil_image", "_image"):
            candidate = getattr(image_obj, attr, None)
            if isinstance(candidate, Image.Image):
                return candidate

        # If raw bytes are available, try to decode them
        for attr in ("data", "bytes", "content"):
            raw = getattr(image_obj, attr, None)
            if raw:
                try:
                    return Image.open(io.BytesIO(raw))
                except Exception:
                    continue

        return None


NODE_CLASS_MAPPINGS = {
    "JMGeminiFlashImageWithTexts": JMGeminiFlashImageWithTexts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMGeminiFlashImageWithTexts": "JM Gemini Flash Image With Texts"
}
