"""
Convert a single eval-format sample (question, options, answer, image_path, index)
to a fixretina parquet row for RL held-out evaluation.
Matches the schema produced by visualprobe2fixretina.py (image = overview, extra_info.seed_img = original).
"""
import math
import os
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

# Image constants (same as visualprobe2fixretina)
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_VIEW_PIXELS = (28 * 16) ** 2

NEW_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"focus","description":"Request a high-resolution local region of the first image and zoom in","parameters":{"type":"object","properties":{"bboxes":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"array","items":{"type":"integer"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to crop, as [x1, y1, x2, y2] in ABSOLUTE PIXEL COORDINATES of the first image."},"description":"A list of bounding boxes to zoom in on. You can request 1-3 bboxes at a turn."}},"required":["bboxes"]}}}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "focus", "arguments": {"bboxes": [[10, 20, 100, 200]]}}  
</tool_call>
"""

USER_PROMPT = """<image>
Think carefully before answering. If the current view is not enough (e.g., the image is too large/unclear/complicated)  or if a previous zoom-in was incorrect, call **image_zoom_in** to gather better evidence and do not give <answer> in the same step. 
Keep track of the original question across multiple steps and always make the zoom-in region highly relevant to the question. 
Only when you are certain about the final answer and no further tool call is needed, then output:
<answer>...</answer>.
**Format stricly as either:**
<think>...</think>
<tool_call>...</tool_call>
**OR**
<think>...</think><answer></answer>
Here is the question:
"""
USER_PROMPT="""<image>\nThink first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question:"""


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(int(height / beta), factor)
        w_bar = _floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(int(height * beta), factor)
        w_bar = _ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar


def _constrain_image_size(
    img: Image.Image, max_pixels: int = MAX_VIEW_PIXELS
) -> Tuple[Image.Image, Tuple[float, float]]:
    if max_pixels is None:
        return img, (1.0, 1.0)
    w, h = img.size
    target_h, target_w = _smart_resize(
        h, w, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=max_pixels
    )
    if target_h == h and target_w == w:
        return img, (1.0, 1.0)
    resized = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    scale_x, scale_y = target_w / w, target_h / h
    return resized, (scale_x, scale_y)


def _pil2bytes(pil_image: Image.Image) -> bytes:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    with BytesIO() as buffer:
        pil_image.save(buffer, format="jpeg")
        return buffer.getvalue()


def _build_prompt(question: str, options: Optional[List[str]]) -> np.ndarray:
    user_content = USER_PROMPT + question
    if options:
        choices_str = ", ".join([f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)])
        user_content += f" choices: {choices_str}"
    messages = [
        {"role": "system", "content": NEW_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return np.array(messages, dtype=object)


def sample_to_fixretina_row(
    sample: Dict[str, Any],
    data_source: str,
    ability: str = "vqa",
) -> Dict[str, Any]:
    """
    Convert one eval-format sample to a fixretina parquet row.

    sample must have: image_path, question, answer, index; optional: options (list).
    """
    image_path = sample["image_path"]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    original_pil = Image.open(image_path).convert("RGB")
    overview_pil, scale = _constrain_image_size(original_pil, MAX_VIEW_PIXELS)
    # pixel_budget = overview_pil.size[0] * overview_pil.size[1]
    pixel_budget = MAX_VIEW_PIXELS

    question = sample["question"]
    options = sample.get("options") or []
    if options is not None and len(options) == 0:
        options = None
    answer_raw = sample["answer"]
    answer_list = [answer_raw] if isinstance(answer_raw, str) else list(answer_raw)
    index = sample.get("index", "")
    if index is not None and not isinstance(index, str):
        index = str(index)
    index = index or ""

    overview_bytes = _pil2bytes(overview_pil)
    original_bytes = _pil2bytes(original_pil)

    extra_info = {
        "answer": answer_list,
        "index": index,
        "options": options,
        "question": question,
        "seed_img": {"bytes": original_bytes, "scale": scale},
        "split": "eval",
        "orig_path": image_path,
        "pixel_budget_per_image": pixel_budget,
    }

    reward_model = {
        "ground_truth": answer_list,
        "style": "multiple_choice" if options else "free_form",
        "ground_truth_bboxes": None,
        "question": question,
        "path": None,
    }

    prompt = _build_prompt(question, options)

    row = {
        "prompt": prompt,
        "images": np.array([{"bytes": overview_bytes, "path": None}], dtype=object),
        "ability": ability,
        "env_name": "fixretina",
        "data_source": data_source,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }
    return row
