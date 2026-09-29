import os
import re
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any

def _safe_parse_tool_call(s: str) -> dict:
    """Parse tool-call JSON inside <tool_call>...</tool_call>."""
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        import json_repair
        return json_repair.loads(s)
    except Exception:
        pass
    import ast
    return ast.literal_eval(s)


def convert_treevgr_to_llamafactory_format(
    record: Dict[str, Any],
    source_image_base: str = "/Volumes/Media/vlmpaper/data/sft/TreeVGR-SFT-35K/llava_next_raw_format",
    output_image_base: str = "/Volumes/Media/vlmpaper/data/sft/TreeVGR-SFT-35K",
    include_seed_image: bool = True
) -> Dict[str, Any]:
    """
    Convert TreeVGR rewritten format to FixRetina SFT training format.
    
    Args:
        record: Input record in TreeVGR format
        source_image_base: Base directory for source images
        output_image_base: Base directory for saving cropped images
        include_seed_image: Whether to include 'seed_image' field
    
    Returns:
        Converted record in FixRetina SFT format
    """
    # Get original image path
    orig_img_rel = record['images'][0] # images/coco/train2017/000000336353.jpg
    orig_img_path = os.path.join(source_image_base, orig_img_rel)
    
    if not os.path.exists(orig_img_path):
        raise FileNotFoundError(f"Source image not found: {orig_img_path}")
    
    # Load original image
    orig_img = Image.open(orig_img_path).convert("RGB")
    w, h = orig_img.size
    
    # Extract directory structure and filename from orig_img_rel
    # orig_img_rel: images/coco/train2017/000000336353.jpg
    # -> dir_part: images/coco/train2017/
    # -> filename: 000000336353.jpg
    orig_img_path_obj = Path(orig_img_rel)
    dir_part = str(orig_img_path_obj.parent)  # images/coco/train2017
    filename_stem = orig_img_path_obj.stem  # 000000336353
    
    # Prepare output directory structure (preserve original directory structure)
    output_dir_full = os.path.join(output_image_base, dir_part)
    os.makedirs(output_dir_full, exist_ok=True)
    
    # Save original image to output dir (first image in the list)
    orig_output_rel = f"{dir_part}/{filename_stem}-0.jpg"
    orig_output_path = os.path.join(output_image_base, orig_output_rel)
    orig_img.save(orig_output_path)
    
    # List to store all image paths (overview + crops) in order
    output_images = [orig_output_rel]
    
    # Parse conversations and extract bboxes, save crops
    conversations_in = record['conversations']
    crop_idx = 1
    
    for msg_idx, msg in enumerate(conversations_in):
        if msg.get('from') != 'gpt':
            continue
        val = msg.get('value', '')
        if '<tool_call>' not in val:
            continue
        
        # Extract tool_call
        m = re.search(r'<tool_call>(.*?)</tool_call>', val, re.DOTALL)
        if not m:
            continue
        
        tc = _safe_parse_tool_call(m.group(1).strip())
        bboxes = tc.get('arguments', {}).get('bboxes', [])
        
        # For each bbox, crop and save
        for bbox in bboxes:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            # Clamp
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            
            if x2 <= x1 or y2 <= y1:
                # 无效 bbox：创建 1x1 占位图，确保 image 数量匹配
                placeholder = Image.new('RGB', (224, 224), color='black')
                crop_rel = f"{dir_part}/{filename_stem}-{crop_idx}.jpg"
                crop_path = os.path.join(output_image_base, crop_rel)
                placeholder.save(crop_path)
                output_images.append(crop_rel)
                crop_idx += 1
                continue
            
            crop = orig_img.crop((x1, y1, x2, y2))
            crop_rel = f"{dir_part}/{filename_stem}-{crop_idx}.jpg"
            crop_path = os.path.join(output_image_base, crop_rel)
            crop.save(crop_path)
            
            output_images.append(crop_rel)
            crop_idx += 1
    
    # Build new conversations with system prompt from original record
    system_prompt = record.get('system', '')
    new_conversations = [
        {
            "from": "system",
            "value": system_prompt
        }
    ]
    
    # Add original conversations (human/gpt turns)
    new_conversations.extend(conversations_in)
    
    # Build output record
    output = {
        "conversations": new_conversations,
        "images": output_images,
    }
    
    # Optional: add seed_image field (preserve original path structure)
    if include_seed_image:
        output["seed_image"] = orig_img_rel  # Keep original: images/coco/train2017/000000336353.jpg
    
    # Optional: preserve other useful fields
    for key in ['answer', 'question', 'options', 'rewriter']:
        if key in record:
            output[key] = record[key]

    # ===== VALIDATION: Check <image> count matches images list =====
    expected_image_count = len(output_images)
    actual_image_count = 0
    
    # Count <image> tags in all conversations
    for conv in new_conversations:
        if conv.get('from') in ['human', 'user']:
            actual_image_count += conv.get('value', '').count('<image>')
    
    if actual_image_count != expected_image_count:
        raise ValueError(
            f"Image count mismatch! "
            f"Expected {expected_image_count} images in list, "
            f"but found {actual_image_count} <image> tags in conversations. "
            f"Image: {orig_img_rel}"
        )
    
    return output
    


# Example usage:
if __name__ == "__main__":
    input_file = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K/TreeVGR-SFT-35K_rewrited.json"
    output_file = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K/treevgr_rewrited_llamafacotry.json"
    with open(input_file, "r") as f:
        data = json.load(f)
    
    sample = data[2000]
    import pprint
    pprint.pprint(sample)
    print("-"*100)
    converted = convert_treevgr_to_llamafactory_format(
        record = sample,
        source_image_base = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K/llava_next_raw_format",
        output_image_base = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K",
        include_seed_image = True
    )   
    
    # Print result
    pprint.pprint(converted)
    
    # Save to JSON
    converted_data = []
    for record in data: 
        try:
            converted = convert_treevgr_to_llamafactory_format(
                record=record,
                source_image_base = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K/llava_next_raw_format",
                output_image_base = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K",
                include_seed_image = True
            )
            converted_data.append(converted)
        except Exception as e:
            print(f"Error converting record: {e}")
            continue
    
    with open(output_file, "w") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(converted_data)} records to {output_file}")