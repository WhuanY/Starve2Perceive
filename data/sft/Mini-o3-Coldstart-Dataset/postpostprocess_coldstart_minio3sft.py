import os
import re
import json
import ast
import math
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image
from tqdm import tqdm


# ===================== Fix-Retina budget =====================
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_VIEW_PIXELS = 16 * 16 * 28 * 28  # 200704


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_VIEW_PIXELS,
) -> Tuple[int, int]:
    """
    Same idea as `vlmpaper/eval/eval_w_tool/inference_agent/image_utils.py::smart_resize`,
    but with our MAX_VIEW_PIXELS budget as default.
    Returns (new_h, new_w).
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)

    return h_bar, w_bar


def constrain_image_size(img: Image.Image, max_pixels: int = MAX_VIEW_PIXELS) -> Tuple[Image.Image, Tuple[float, float]]:
    """
    Returns (resized_img, (scale_x, scale_y)) where
    scale_x = new_w / orig_w, scale_y = new_h / orig_h.
    """
    if max_pixels is None:
        return img, (1.0, 1.0)

    orig_w, orig_h = img.size
    target_h, target_w = smart_resize(orig_h, orig_w, max_pixels=max_pixels)

    if target_h == orig_h and target_w == orig_w:
        return img, (1.0, 1.0)

    resized = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return resized, (target_w / orig_w, target_h / orig_h)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_jpg(img: Image.Image, path: str, quality: int = 95) -> None:
    _ensure_dir(os.path.dirname(path))
    img.save(path, format="JPEG", quality=quality)


def _images_dir_to_bpix(images_dir: str) -> str:
    """
    Convert .../images/<case_id>/  ->  .../images_bpix/<case_id>/
    """
    # safest: replace only one occurrence of "/images/" segment
    return images_dir.replace(f"{os.sep}images{os.sep}", f"{os.sep}images_bpix{os.sep}", 1)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_tool_call_payload(payload: str) -> Optional[Dict[str, Any]]:
    """
    payload is expected to be a JSON object string like:
      {"name":"focus","arguments":{"bboxes":[[...]]}}
    Some samples may be not strict JSON -> fallback to literal_eval.
    """
    payload = payload.strip()
    try:
        return json.loads(payload)
    except Exception:
        try:
            obj = ast.literal_eval(payload)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _dump_tool_call_payload(obj: Dict[str, Any]) -> str:
    """
    Write back a strict JSON string (stable & parseable).
    """
    return json.dumps(obj, ensure_ascii=False)


def _bbox_to_overview(bbox_orig: List[int], scale_x: float, scale_y: float, ov_w: int, ov_h: int) -> List[int]:
    x1, y1, x2, y2 = bbox_orig
    x1o = int(round(x1 * scale_x))
    y1o = int(round(y1 * scale_y))
    x2o = int(round(x2 * scale_x))
    y2o = int(round(y2 * scale_y))

    # clamp into overview bounds
    x1o = max(0, min(x1o, ov_w))
    x2o = max(0, min(x2o, ov_w))
    y1o = max(0, min(y1o, ov_h))
    y2o = max(0, min(y2o, ov_h))

    # keep (x1<x2, y1<y2) if rounding collapses
    if x2o <= x1o:
        x2o = min(ov_w, x1o + 1)
    if y2o <= y1o:
        y2o = min(ov_h, y1o + 1)

    return [x1o, y1o, x2o, y2o]


def _regex_for_bbox_bracketed(b: List[int]) -> re.Pattern:
    # matches "[ 0 ,0, 1008 , 750 ]" variants
    x1, y1, x2, y2 = b
    pat = rf"\[\s*{x1}\s*,\s*{y1}\s*,\s*{x2}\s*,\s*{y2}\s*\]"
    return re.compile(pat)


def _regex_for_bbox_unbracketed(b: List[int]) -> re.Pattern:
    # matches "0, 0, 1008, 750" variants
    x1, y1, x2, y2 = b
    pat = rf"{x1}\s*,\s*{y1}\s*,\s*{x2}\s*,\s*{y2}"
    return re.compile(pat)


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _update_coordinates_in_think_sections(text: str, bboxes_orig: List[List[int]], bboxes_ov: List[List[int]]) -> str:
    """
    Only rewrite inside <think>...</think> blocks (avoid accidentally touching other regions).
    """

    def _rewrite_think_block(block: str) -> str:
        updated = block
        for bo, bn in zip(bboxes_orig, bboxes_ov):
            updated = _regex_for_bbox_bracketed(bo).sub(f"[{bn[0]}, {bn[1]}, {bn[2]}, {bn[3]}]", updated)
            updated = _regex_for_bbox_unbracketed(bo).sub(f"{bn[0]}, {bn[1]}, {bn[2]}, {bn[3]}", updated)
        return updated

    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        return f"<think>{_rewrite_think_block(inner)}</think>"

    return _THINK_RE.sub(_repl, text)


def reconstruct_img_and_bbox(item: Dict[str, Any], max_view_pixels: int = MAX_VIEW_PIXELS) -> Dict[str, Any]:
    """
    Input item format:
      {
        "conversations": [...],
        "images": [original_image_path, observation_1_path, ...]
      }
    Output:
      same schema, but images moved to images_bpix/, tool_call bbox rewritten to overview pixel space,
      and observation images regenerated from original crops then constrained to max_view_pixels.
    """
    new_item = deepcopy(item)

    # ---- 1) load original image
    orig_path = new_item["images"][0]
    # Preserve the uncompressed original image path for later reference.
    # This is the "seed" image before any fix-retina style resizing/cropping.
    new_item["seed_img"] = orig_path
    orig_img = Image.open(orig_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # ---- 2) overview & scale
    overview_img, (scale_x, scale_y) = constrain_image_size(orig_img, max_pixels=max_view_pixels)
    ov_w, ov_h = overview_img.size

    # ---- 3) output directory
    orig_dir = os.path.dirname(orig_path)  # .../images/<case_id>
    out_dir = _images_dir_to_bpix(orig_dir)  # .../images_bpix/<case_id>
    _ensure_dir(out_dir)

    # ---- 4) save overview
    # out_overview_path = os.path.join(out_dir, "original_image.jpg")
    out_overview_path = os.path.join(out_dir, "overview_image.jpg")
    _save_jpg(overview_img, out_overview_path)

    # ---- 5) walk conversations: rewrite tool_call bbox + think text; regenerate observations
    obs_paths: List[str] = []
    obs_idx = 1

    new_convs = []
    for conv in new_item.get("conversations", []):
        if conv.get("from") != "gpt":
            new_convs.append(conv)
            continue

        txt = conv.get("value", "")
        tool_calls = list(_TOOL_CALL_RE.finditer(txt))
        if not tool_calls:
            new_convs.append(conv)
            continue

        # We rebuild txt progressively
        rebuilt = []
        last_end = 0

        for m in tool_calls:
            payload_raw = m.group(1)
            tool_obj = _parse_tool_call_payload(payload_raw)

            # append prefix
            rebuilt.append(txt[last_end:m.start()])

            if (
                tool_obj is None
                or not isinstance(tool_obj, dict)
                or tool_obj.get("name") != "focus"
                or "arguments" not in tool_obj
                or not isinstance(tool_obj["arguments"], dict)
                or "bboxes" not in tool_obj["arguments"]
            ):
                # keep original tool_call
                rebuilt.append(m.group(0))
                last_end = m.end()
                continue

            bboxes_orig = tool_obj["arguments"]["bboxes"]
            if not isinstance(bboxes_orig, list):
                rebuilt.append(m.group(0))
                last_end = m.end()
                continue

            # enforce int bbox lists
            cleaned_orig: List[List[int]] = []
            for b in bboxes_orig:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    cleaned_orig.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])

            if not cleaned_orig:
                rebuilt.append(m.group(0))
                last_end = m.end()
                continue

            # map to overview pixel space
            bboxes_ov = [_bbox_to_overview(b, scale_x, scale_y, ov_w, ov_h) for b in cleaned_orig]
            tool_obj["arguments"]["bboxes"] = bboxes_ov

            # regenerate observations using ORIGINAL bbox (crop in original pixel space)
            for b in cleaned_orig:
                x1, y1, x2, y2 = b
                # clamp to original just in case
                x1 = max(0, min(x1, orig_w))
                x2 = max(0, min(x2, orig_w))
                y1 = max(0, min(y1, orig_h))
                y2 = max(0, min(y2, orig_h))
                if x2 <= x1 or y2 <= y1:
                    # skip invalid
                    # continue
                    raise ValueError(f"Invalid bbox: {b}")

                crop = orig_img.crop((x1, y1, x2, y2))
                obs_img, _ = constrain_image_size(crop, max_pixels=max_view_pixels)

                out_obs_path = os.path.join(out_dir, f"observation_{obs_idx}.jpg")
                _save_jpg(obs_img, out_obs_path)
                obs_paths.append(out_obs_path)
                obs_idx += 1

            # write back tool_call with updated JSON
            rebuilt.append("<tool_call>\n" + _dump_tool_call_payload(tool_obj) + "\n</tool_call>")

            # update think text (inside the whole message) based on this tool_call mapping
            # NOTE: do this AFTER we rebuild tool_call, but we can apply on the suffix later.
            # We will apply once at the end using all mappings found in this message.
            last_end = m.end()

        # append suffix
        rebuilt.append(txt[last_end:])
        new_txt = "".join(rebuilt)

        # second pass: update coordinates inside <think> sections for ALL tool calls in this message
        # collect mappings again from rewritten tool_calls: we need orig->new; we can re-parse from original `tool_calls`.
        # simplest: redo parse from original txt and recompute mapping deterministically
        all_orig: List[List[int]] = []
        all_new: List[List[int]] = []
        for m in tool_calls:
            tool_obj = _parse_tool_call_payload(m.group(1))
            if tool_obj and tool_obj.get("name") == "focus":
                bboxes = tool_obj.get("arguments", {}).get("bboxes", [])
                if isinstance(bboxes, list):
                    for b in bboxes:
                        if isinstance(b, (list, tuple)) and len(b) == 4:
                            bo = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                            bn = _bbox_to_overview(bo, scale_x, scale_y, ov_w, ov_h)
                            all_orig.append(bo)
                            all_new.append(bn)

        new_txt = _update_coordinates_in_think_sections(new_txt, all_orig, all_new)

        new_conv = dict(conv)
        new_conv["value"] = new_txt
        new_convs.append(new_conv)

    # If we failed to regenerate any obs (edge case), fall back to compress existing observation images.
    if not obs_paths and len(new_item.get("images", [])) > 1:
        for i, old_obs_path in enumerate(new_item["images"][1:], start=1):
            try:
                img = Image.open(old_obs_path).convert("RGB")
                obs_img, _ = constrain_image_size(img, max_pixels=max_view_pixels)
                out_obs_path = os.path.join(out_dir, f"observation_{i}.jpg")
                _save_jpg(obs_img, out_obs_path)
                obs_paths.append(out_obs_path)
            except Exception:
                continue

    new_item["conversations"] = new_convs
    new_item["images"] = [out_overview_path] + obs_paths
    return new_item


if __name__ == "__main__":
    in_json = "Mini-o3-Coldstart-Dataset-wo-compression.json"
    out_json = "Mini-o3-Coldstart-Dataset-w-compression.json"

    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in tqdm(data, total=len(data), desc="Processing items"):
        new_data.append(reconstruct_img_and_bbox(item))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)