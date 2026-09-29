import math
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
import re
import json
import io

# Image processing constants (from fixretina)
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO=200
MAX_VIEW_PIXELS = (28 * 16) ** 2
ERROR_INFO_MULTI_TURN_PROMPT="Please analyze the error information obtained from the function tool and adjust your response. Continue your reasoning process inside <think> and </think>."


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    """
    Smart resize image dimensions to fit within pixel constraints.
    
    Args:
        height: Original height
        width: Original width
        factor: Rounding factor for dimensions
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
        
    Returns:
        Tuple of (new_height, new_width)
    """
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
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
    Constrain the image size to the maximum pixel budget.
    
    Returns:
        (resized_img, (scale_x, scale_y)): Tuple of (resized_img, (scale_x, scale_y))
    """
    if max_pixels is None:
        return img, (1.0, 1.0)
        
    w, h = img.size
    
    target_h, target_w = smart_resize(h, w,
                                      factor=IMAGE_FACTOR,
                                      min_pixels=MIN_PIXELS,
                                      max_pixels=max_pixels)
    
    if target_h == h and target_w == w:
        return img, (1.0, 1.0)
    
    resized_img = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    
    scale_x, scale_y = target_w / w, target_h / h
    
    return resized_img, (scale_x, scale_y)


class FixRetinaTool(ToolBase):
    """
    FixRetina tool for multi-bbox image focusing with coordinate mapping.
    
    This tool implements the fixretina workflow:
    - Supports multiple bboxes (1-3 per call)
    - Maps coordinates from overview image to original image
    - Constrains cropped images to MAX_VIEW_PIXELS
    """
    name = "fixretina"
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(name=self.name)
        self.chatml_history = []
        self.original_image = None  # Original PIL image
        self.overview_scale = (1.0, 1.0)  # (scale_x, scale_y) from overview to original
        self.multi_modal_data = None  # Current multi-modal data
        # self.pixel_budget_per_image = 16 * 16 * 28 * 28 # default value
        
    def extract_answer(self, action_string: str) -> Optional[str]:
        """Extract answer from action string."""
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Optional[str]:
        """Extract tool call from action string."""
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def zoom_in_with_coordinate_mapping(self, selected_bbox: List[int]) -> Image.Image:
        """
        Zoom in on a region using coordinate mapping from overview to original image.
        
        Args:
            selected_bbox: Bounding box [x1, y1, x2, y2] in overview image coordinates
            
        Returns:
            Cropped and constrained PIL Image
        """
        if self.original_image is None:
            raise ValueError("Original image not set. Call reset() first.")
        
        orig_img = self.original_image
        orig_w, orig_h = orig_img.size
        scale_x, scale_y = self.overview_scale
        
        selected_x1, selected_y1, selected_x2, selected_y2 = selected_bbox
        
        # 1. Map back to the original image coordinates
        raw_x1 = int(round(selected_x1 / scale_x))
        raw_y1 = int(round(selected_y1 / scale_y))
        raw_x2 = int(round(selected_x2 / scale_x))
        raw_y2 = int(round(selected_y2 / scale_y))
        
        # 2. Clamping - ensure coordinates are within bounds
        final_x1 = max(0, min(raw_x1, orig_w))
        final_y1 = max(0, min(raw_y1, orig_h))
        final_x2 = max(0, min(raw_x2, orig_w))
        final_y2 = max(0, min(raw_y2, orig_h))
        
        # 3. Check validity of coordinates
        if final_x1 >= final_x2 or final_y1 >= final_y2:
            error_msg = (f"Invalid crop coordinates: bbox={selected_bbox}, "
                 f"scale=({scale_x:.3f}, {scale_y:.3f}), "
                 f"final_coords=({final_x1}, {final_y1}, {final_x2}, {final_y2}), "
                 f"orig_size=({orig_w}, {orig_h})"
            )
            print(error_msg, flush=True)
            raise ValueError(error_msg)
        
        # 4. Crop from original image
        raw_crop = orig_img.crop((final_x1, final_y1, final_x2, final_y2))
        
        # 5. Constrain size 
        view_img, _ = constrain_image_size(raw_crop, max_pixels=self.pixel_budget_per_image)
        
        return view_img

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the fixretina tool functionality.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            tuple: (observation, reward, done, info)
                - observation: Dict with prompt and multi_modal_data
                - reward: float (0.0 for tool calls)
                - done: bool (True if answer found, False otherwise)
                - info: Dict with status and additional info
        """
        # Check for answer first
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        
        # Extract tool call
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}
        
        try:
            tool_call = json.loads(action.strip())
        except Exception as e:
            error_info = f"ERROR occurs during parsing tool call. Error Information: Invalid tool call format: {action.strip()}. Error: {e}.\n"
            obs = "\n<|im_start|>user\n" + error_info + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
            info = {"error": str(e), "status": "failed"}
            return obs, 0.0, False, info  # Continue reasoning, let model recover from error
        
        try:
            tool_name = tool_call.get("name", "")
            if tool_name != "focus":
                raise ValueError(f"Expected tool name 'focus', got '{tool_name}'")
            
            args = tool_call.get("arguments", {})
            bboxes = args.get("bboxes", [])
            
            if not isinstance(bboxes, list) or len(bboxes) == 0:
                raise ValueError(f"Invalid bboxes: expected non-empty list, got {bboxes}")
            
            if len(bboxes) > 3:
                raise ValueError(f"Too many bboxes: expected 1-3, got {len(bboxes)}")
            
            # Process each bbox
            cropped_images = []
            for i, bbox in enumerate(bboxes):
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Invalid bbox format at index {i}: expected [x1, y1, x2, y2], got {bbox}")
                
                try:
                    cropped_img = self.zoom_in_with_coordinate_mapping(bbox)
                    c_w, c_h = cropped_img.size 
                    if min(c_w, c_h) <= 0 or max(c_w, c_h) / min(c_w, c_h) > MAX_RATIO:
                        raise ValueError(
                            f"Post-resize aspect ratio too extreme for bbox {bbox}: "
                            f"size=({c_w}, {c_h}), AR={max(c_w, c_h) / max(min(c_w, c_h), 1):.1f}"
                        )
                    cropped_images.append(cropped_img)
                except Exception as e:
                    print(f"[DEBUG] Failed to process bbox {i}: {bbox}, error: {e}")
                    # Continue with other bboxes even if one fails
                    continue

            if len(cropped_images) == 0:
                raise ValueError("All bboxes failed to process")

            # Prepare observation with all cropped images
            obs = {
                "prompt": "\n<|im_start|>user\n" + "<tool_response>" + "<image>" * len(cropped_images) + "</tool_response>" + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": cropped_images}
            }
            
            reward = 0.0
            done = False
            info = {
                "status": "success",
                "tool_used": tool_name,
                "num_bboxes": len(bboxes),
                "num_cropped": len(cropped_images)
            }
            
            print(f'[DEBUG] SUCCESS ACTION {action_string=}, processed {len(cropped_images)} bboxes')
            return obs, reward, done, info
            
        except Exception as e:
            # Return error observation
            print(f'[DEBUG] Execute WRONG - {str(e)} {action_string=}')
            error_info = f"ERROR occurs during focus operation. Error Information: {str(e)}.\n"
            obs = "\n<|im_start|>user\n" + error_info + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """
        Reset the tool state with new data.
        
        Args:
            raw_prompt: Original prompt text
            multi_modal_data: overview image that first passed by dataset then passed by qwen processor
            origin_multi_modal_data: overview image passed by dataset
        """
        self.chatml_history = raw_prompt
        # print('[DEBUG] reset fixretina', multi_modal_data.keys())
        self.multi_modal_data = multi_modal_data
        
        assert 'image' in origin_multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(origin_multi_modal_data['image']) > 0, f'[ERROR] {origin_multi_modal_data["image"]=}'
        
        seed_img = multi_modal_data.pop('seed_img', None)
        pixel_budget_per_image = multi_modal_data.pop('pixel_budget_per_image', None)
        assert seed_img is not None, f'[ERROR] {seed_img=}'
        assert pixel_budget_per_image is not None, f'[ERROR] {pixel_budget_per_image=}'

        seed_img_bytes = seed_img.pop('bytes', None)
        assert seed_img_bytes is not None, f'[ERROR] {seed_img_bytes=}'
        # convert bytes to pilimage
        self.original_image = Image.open(io.BytesIO(seed_img_bytes))
        del seed_img_bytes
        scale_x, scale_y = seed_img.pop('scale') # 数据预处理阶段算出来的scale
        
        orig_w, orig_h = self.original_image.size
        
        # Calculate scale if overview image exists
        if multi_modal_data and 'image' in multi_modal_data and len(multi_modal_data['image']) > 0:
            overview_img = multi_modal_data['image'][0]
            # self.pixel_budget_per_image = multi_modal_data.pop('pixel_budget_per_image', None)
            # assert self.pixel_budget_per_image is not None, f'[ERROR] {self.pixel_budget_per_image=}'
            ov_w, ov_h = overview_img.size
            # assert ov_w * ov_h == pixel_budget_per_image, f'[ERROR] {ov_w * ov_h=} != {pixel_budget_per_image}' # 检查一下pixel_budget_per_image是否一致，如果不一致就是哪里错了
            self.pixel_budget_per_image = pixel_budget_per_image
            # Calculate scale from overview to original
            overview_scale = (ov_w / orig_w, ov_h / orig_h)
            if pixel_budget_per_image < MAX_VIEW_PIXELS:
                assert overview_scale[0] == scale_x and overview_scale[1] == scale_y, f'[ERROR] {overview_scale=} != {scale_x, scale_y}' # 检查一下scale是否一致，如果不一致就是哪里错了
            self.overview_scale = (ov_w / orig_w, ov_h / orig_h)
        else:
            # No overview, use identity scale
            self.overview_scale = (1.0, 1.0)

        
        print(f'[DEBUG] FixRetinaTool reset: orig_size=({orig_w}, {orig_h}), scale={self.overview_scale}, ov_size=({ov_w}, {ov_h}), {pixel_budget_per_image=}')
