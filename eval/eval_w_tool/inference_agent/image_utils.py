"""
Image processing utilities for tool calling.
"""
import base64
import math
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Image processing constants
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

# ---- Added for Fix Retina ----
# 28: Pixel Length for a Patch of QwenVL Tokenizer; 
# 16: Draw inspiration from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" 
# Under this setting and by proper processing, the maximum tokens viewed by qwenvl will not exceed 256. 
# Such setting is heuristic and may not be optimal.

MAX_VIEW_PIXELS = (28 * 16) ** 2 

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    """Encode a PIL Image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

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
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
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


# ---- Added for Fix Retina ----
def count_tokens(img_input=None, width=None, height=None):
    """
    Count the number of tokens for an image using Qwen-VL tokenizer logic.
    """
    # If width and height are provided, use them directly
    if width is not None and height is not None:
        pass  # Use provided dimensions
    # If img_input is a string (path), load the image
    elif isinstance(img_input, str):
        img = Image.open(img_input)
        width, height = img.size
    # If img_input is a PIL Image
    elif hasattr(img_input, 'size'):  # Check for PIL Image (duck typing)
        width, height = img_input.size
    else:
        raise ValueError(
            "Must provide either:\n"
            "  - img_input as str (file path)\n"
            "  - img_input as PIL.Image.Image\n"
            "  - both width and height parameters"
        )
    
    # Calculate tokens using smart_resize logic
    resized_h, resized_w = smart_resize(height, width)
    image_tokens = (resized_h // IMAGE_FACTOR) * (resized_w // IMAGE_FACTOR)
    
    return image_tokens


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


def zoom_in_image_w_coordinate_mapping(selected_bbox, traj_info):
    """
    Based on the selected bbox, zoom in the image and return the zoomed in image.
    """
    orig_img = traj_info['original_pil_img']
    orig_w, orig_h = orig_img.size 
    scale_x, scale_y = traj_info['scale']

    selected_x1, selected_y1, selected_x2, selected_y2 = selected_bbox

    # 1. Map back to the original image coordinates
    raw_x1 = int(round(selected_x1 / scale_x))
    raw_y1 = int(round(selected_y1 / scale_y))
    raw_x2 = int(round(selected_x2 / scale_x))
    raw_y2 = int(round(selected_y2 / scale_y))

    # 2. Clamping. 
    # Ensure the coordinates are not less than 0 and not greater than the original image size.
    final_x1 = max(0, min(raw_x1, orig_w))
    final_y1 = max(0, min(raw_y1, orig_h))
    final_x2 = max(0, min(raw_x2, orig_w))
    final_y2 = max(0, min(raw_y2, orig_h))

    # 3. Check the validity of the coordinates.
    if final_x1 >= final_x2 or final_y1 >= final_y2:
        logger.warning(
            f"Invalid crop coordinates: bbox={selected_bbox}, "
            f"scale=({scale_x:.3f}, {scale_y:.3f}), "
            f"final_coords=({final_x1}, {final_y1}, {final_x2}, {final_y2}), "
            f"orig_size=({orig_w}, {orig_h})"
        )
        print(f"Invalid crop coordinates: bbox={selected_bbox}, scale=({scale_x:.3f}, {scale_y:.3f}), final_coords=({final_x1}, {final_y1}, {final_x2}, {final_y2}), orig_size=({orig_w}, {orig_h})", flush=True)
        # Return a black square to prevent program crash.
        return Image.new('RGB', (224, 224), (0, 0, 0))

    raw_crop = orig_img.crop((final_x1, final_y1, final_x2, final_y2))

    # 5. NOTE: Essential for our design. 
    view_img, _ = constrain_image_size(raw_crop, max_pixels=MAX_VIEW_PIXELS)
    logger.info(f"Successfully zoomed in the image: bbox={selected_bbox}, scale=({scale_x:.3f}, {scale_y:.3f}), final_coords=({final_x1}, {final_y1}, {final_x2}, {final_y2}), orig_size=({orig_w}, {orig_h})")
    return view_img


def focus(bboxes: List[Tuple[int, int, int, int]], traj_info: Dict[str, Any]) -> List[str]:
    """
    Focus on the specific regions of the image.
    """
    encoded_imgs = []
    cropped_tokens = []
    for bbox in bboxes:
        cropped_img = zoom_in_image_w_coordinate_mapping(bbox, traj_info)
        encoded_img = encode_pil_image_to_base64(cropped_img)
        cropped_tokens.append(count_tokens(cropped_img))
        encoded_imgs.append(encoded_img)
    
    traj_info['perf_stats']['crop_tokens'].append(cropped_tokens) 
    return encoded_imgs

