"""
Convert DeepEyes parquet rows into FixRetina RL format.

Target format follows:
- `vlmpaper/data/rl/fixretina_rl/train/note.md`
- `vlmpaper/data/rl/ViRL8k/virl8k_to_fixretina.py`
- `vlmpaper/data/rl/TreeVGR-RL-37K/treevgr37k_to_fixretina.py`
"""

import argparse
import io
import math
import multiprocessing as mp
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from pandas.io import parsers
from tqdm import tqdm


SYSTEM_PROMPT = """You are a helpful assistant.

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

USER_PROMPT_PREFIX = (
    "<image>\n"
    "Think first, call **focus** if needed, then answer if you are confident. "
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  "
    "You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. "
    "Here is the question:\n"
)

# Qwen-VL patch/tokenization constants (copied from referenced converters)
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
# FixRetina overview budget: (28 * 16)^2 == 448^2 pixels
MAX_VIEW_PIXELS = (28 * 16) ** 2
MAX_RATIO = 200

def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize_downscale_only(height: int, width: int, *, factor: int, max_pixels: int) -> Tuple[int, int]:
    """Only downscale so that (h*w) <= max_pixels; keep aspect ratio; dims multiple of factor."""
    if max(height, width) / max(1, min(height, width)) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}")

    # Round to nearest patch grid
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar <= max_pixels:
        return h_bar, w_bar

    beta = math.sqrt((height * width) / max_pixels)
    h_bar = _floor_by_factor(int(height / beta), factor)
    w_bar = _floor_by_factor(int(width / beta), factor)
    h_bar = max(factor, h_bar)
    w_bar = max(factor, w_bar)
    return h_bar, w_bar


def constrain_image_size(img: Image.Image, *, max_pixels: int = MAX_VIEW_PIXELS) -> Tuple[Image.Image, Tuple[float, float]]:
    """Resize to fit pixel budget; returns resized image and (scale_x, scale_y)."""
    w, h = img.size
    if w * h <= max_pixels:
        return img, (1.0, 1.0)

    target_h, target_w = _smart_resize_downscale_only(h, w, factor=IMAGE_FACTOR, max_pixels=max_pixels)
    if target_h == h and target_w == w:
        return img, (1.0, 1.0)

    resized_img = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    scale_x, scale_y = target_w / w, target_h / h
    return resized_img, (scale_x, scale_y)


def pil2bytes(pil_image: Image.Image) -> bytes:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="jpeg")
        return buffer.getvalue()


def get_first_image_bytes(images: Any) -> bytes:
    if images is None:
        return b""
    if isinstance(images, np.ndarray):
        images = images.tolist()
    if not isinstance(images, list) or len(images) == 0:
        return b""
    img0 = images[0]
    if not isinstance(img0, dict):
        return b""
    return img0.get("bytes", b"") or b""


def get_image_size_from_bytes(img_bytes: bytes) -> Tuple[int, int]:
    if not img_bytes:
        return 0, 0
    img = Image.open(io.BytesIO(img_bytes))
    return img.size  # (w, h)


def filter_records_by_min_pixels(records: List[dict], *, min_pixels: int) -> List[dict]:
    """Keep only records whose first image has > min_pixels pixels."""
    filtered_records: List[dict] = []
    for record in tqdm(records, desc=f"Filtering by image size > {min_pixels}"):
        img_bytes = get_first_image_bytes(record.get("images"))
        w, h = get_image_size_from_bytes(img_bytes)
        if w * h > min_pixels:
            filtered_records.append(record)
    return filtered_records


def build_prompt(question: str, options: Optional[List[str]] = None) -> np.ndarray:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_PREFIX
            + question
            + (
                ("\n" + "\n".join(f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)))
                if options
                else ""
            ),
        },
    ]
    return np.array(messages, dtype=object)


def normalize_to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def split_question_options(question_str: str) -> Tuple[str, Optional[List[str]]]:
    """
    Split a question string into:
    - question: str
    - options: Optional[List[str]] (None if not multiple-choice)
    
    Supported patterns:
    - "... choices: A: ... B: ..."
    - "... Options: (A) ... (B) ..."
    - DeepEyes chart inline:
        "...?A. ...\\nB. ...\\nC. ...\\nD. ..."
    """
    if question_str is None:
        return "", None

    s = str(question_str).replace("<image>", "").strip()

    # 1) Marker-based split (Options: / choices:)
    split_pattern = r"^(.*?)(?:Options:|choices:)\s*(.*)$"
    match = re.match(split_pattern, s, re.IGNORECASE | re.DOTALL)
    if match:
        question = match.group(1).strip()
        options_part = match.group(2).strip()

        options_vgr = re.findall(r"\(([A-Z])\)\s*([^\n]+)", options_part)
        if options_vgr:
            opt_map = {lab: txt.strip() for lab, txt in options_vgr}
            labels = sorted(opt_map.keys())
            return question, [opt_map[l] for l in labels]

        options_std = re.findall(r"([A-Z])\s*:\s*([^\n,]+(?:\n|,|$))", options_part)
        if options_std:
            opt_map = {lab: txt.rstrip(",\n ").strip() for lab, txt in options_std}
            labels = sorted(opt_map.keys())
            return question, [opt_map[l] for l in labels]

        lines = [line.strip() for line in options_part.split("\n") if line.strip()]
        return question, (lines if lines else None)

    # 2) Inline options like A. ... B. ... (no marker, sometimes no newline before A.)
    marker_re = re.compile(r"(?<![A-Z0-9])([A-E])\s*[\.\):]\s*", re.DOTALL)
    matches = list(marker_re.finditer(s))
    if len(matches) >= 2:
        question = s[: matches[0].start()].strip()
        opt_map: Dict[str, str] = {}
        for i, m in enumerate(matches):
            lab = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
            opt_map[lab] = s[start:end].strip()
        labels = sorted(opt_map.keys())
        return question, [opt_map[l] for l in labels]

    return s, None


def normalize_mc_answer(answer: Any) -> Optional[str]:
    """Extract MC label A-E if present; otherwise None."""
    if answer is None:
        return None
    if isinstance(answer, list) and len(answer) == 1:
        answer = answer[0]
    if not isinstance(answer, str):
        answer = str(answer)
    a = answer.strip()
    # Common forms: "B", "(B)", "B.", "Answer: B"
    m = re.search(r"\b([A-E])\b", a)
    if m:
        return m.group(1)
    m = re.match(r"^\(?([A-E])\)?[\.\):]?\s*$", a)
    if m:
        return m.group(1)
    return None


def convert_single_record(record: dict, compression_ratio: Optional[float] = None) -> dict:
    """Convert one row to FixRetina schema."""
    # Load original image from bytes (seed image)
    original_img_bytes = get_first_image_bytes(record.get("images"))
    if not original_img_bytes:
        raise ValueError("missing images[0].bytes")

    original_pil_img = Image.open(io.BytesIO(original_img_bytes)).convert("RGB")
    
    # ----- determine the pixel_budegt_per_image based on the original image size and the compression ratio -----
    max_view_pixels = MAX_VIEW_PIXELS
    if compression_ratio:
        compressed_width = original_pil_img.size[0] * compression_ratio
        compressed_height = original_pil_img.size[1] * compression_ratio
        compressed_max_pixels = compressed_width * compressed_height
        max_view_pixels = min(compressed_max_pixels, MAX_PIXELS) # max_view_pixels is impossible to be larger than original pixels.
        if compression_ratio == 1:
            max_view_pixels = MAX_PIXELS # original setting
    else: # fixretina setting or original setting
        original_width, original_height = original_pil_img.size
        original_pixels = original_width * original_height
        if original_pixels > MAX_VIEW_PIXELS:
            max_view_pixels = MAX_VIEW_PIXELS
        else:
            max_view_pixels = MAX_PIXELS # Fallback to naive think-with-image

    # ----- generate the overview image and calculate the scale -----
    overview_pil_img, scale = constrain_image_size(original_pil_img, max_pixels=max_view_pixels)
    overview_img_bytes = pil2bytes(overview_pil_img)
    pixel_budget_per_image = max_view_pixels

    extra_info_in = record.get("extra_info") or {}
    raw_question_str = extra_info_in.get("question", "")
    if not isinstance(raw_question_str, str) or len(raw_question_str.strip()) == 0:
        raise ValueError("missing extra_info.question")

    question, options = split_question_options(raw_question_str)
    if not isinstance(question, str) or len(question.strip()) == 0:
        raise ValueError("empty question after splitting options")

    answer_in = extra_info_in.get("answer")
    if options is not None:
        ans_lab = normalize_mc_answer(answer_in)
        if ans_lab is None:
            raise ValueError(f"multiple-choice question but cannot parse answer label from {answer_in!r}")
        answer_list: List[Any] = [ans_lab]
    else:
        answer_list = normalize_to_list(answer_in) if answer_in is not None else []

    # extra_info: keep original keys, but enforce FixRetina fields
    extra_info_out = dict(extra_info_in)
    extra_info_out["answer"] = answer_list
    extra_info_out["question"] = question
    extra_info_out["options"] = options  # None for free-form; List[str] for MC
    extra_info_out["seed_img"] = {
        "bytes": original_img_bytes,
        "scale": list(scale),
    }
    extra_info_out['pixel_budget_per_image'] = pixel_budget_per_image

    # reward_model: enrich to match other converters
    reward_model_in = record.get("reward_model") or {}
    ground_truth = reward_model_in.get("ground_truth", answer_list[0] if answer_list else None)
    if options is not None:
        gt_lab = normalize_mc_answer(ground_truth) or (answer_list[0] if answer_list else None)
        ground_truth_list = [gt_lab] if gt_lab is not None else []
        style = "multiple_choice"
    else:
        ground_truth_list = normalize_to_list(ground_truth) if ground_truth is not None else []
        style = "free_form"
    reward_model_out = {
        "ground_truth": ground_truth_list,
        "style": style,
        "ground_truth_bboxes": None,
        "question": question,
        "path": None,
    }

    images_out = np.array([{"bytes": overview_img_bytes, "path": None}], dtype=object)

    out = {
        "data_source": "deepeyes_chart",
        "prompt": build_prompt(question, options),
        "images": images_out,
        "ability": record.get("ability"),
        "env_name": "fixretina",
        "reward_model": reward_model_out,
        "extra_info": extra_info_out,
    }
    return out


def _convert_wrapper(args: Tuple[dict, Optional[float]]) -> Optional[dict]:
    record, compression_ratio = args
    try:
        return convert_single_record(record, compression_ratio=compression_ratio)
    except Exception as e:
        # Keep the script robust; caller can inspect logs.
        print(f"[WARN] failed converting a record: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        type=str,
        default="data_0.1.2_visual_toolbox_v2.parquet",
        help="Input parquet path (DeepEyes format).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data_0.1.2_visual_toolbox_v2_fixretina.parquet",
        help="Output parquet path (FixRetina format).",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=256 * 256,
        help="Filter out images with <= min_pixels.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Multiprocessing workers.",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=None,
        help="Sample number.",
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=None,
        help="Compression ratio for image processing. If 1, uses MAX_PIXELS. If None, uses fixretina logic.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    orig_records = df.to_dict(orient="records")
    print(f"Original records: {len(orig_records)}")

    print(f"Filtering records by image pixels > {args.min_pixels} ...")
    filtered_records = filter_records_by_min_pixels(orig_records, min_pixels=args.min_pixels)
    print(f"Filtered records: {len(filtered_records)}")
    print(f"Removed records: {len(orig_records) - len(filtered_records)}")
    if len(orig_records):
        if args.sample_num:
            import random 
            seed = 42
            random.seed(seed)
            filtered_records = random.sample(filtered_records, args.sample_num)
            print(f"Sampled {args.sample_num} records from {len(filtered_records)}")
        print(f"Retention rate: {len(filtered_records) / len(orig_records) * 100:.2f}%")

    print(f"Converting to FixRetina format with {args.num_workers} workers ...")
    compression_ratio = args.compression_ratio
    if compression_ratio is not None:
        print(f"Using compression_ratio: {compression_ratio}")
    else:
        print("Using default fixretina setting (compression_ratio=None)")
    
    args_list = [(record, compression_ratio) for record in filtered_records]
    with mp.Pool(processes=args.num_workers) as pool:
        converted = list(
            tqdm(pool.imap(_convert_wrapper, args_list, chunksize=32), total=len(filtered_records))
        )

    converted = [r for r in converted if r is not None]
    print(f"Successfully converted: {len(converted)}/{len(filtered_records)}")

    out_df = pd.DataFrame(converted)
    out_df.to_parquet(args.out_path)
    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main()