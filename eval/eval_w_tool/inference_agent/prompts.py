"""
Prompt templates for tool calling and VQA tasks.
"""
from typing import Dict
import json

# System prompt for tool calling
INSTRUCTION_PROMPT_SYSTEM = """You are a helpful assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""

INSTRUCTION_PROMPT_SYSTEM_FIX_RETINA="""You are a helpful assistant.
A user gives a image with a question. Your task is to solve the question based on the **Fixed Retina** constraint:
- MAX_VIEW_PIXELS = 28 * 28 * 16 * 16 pixels.
- The user's image is compressed into an `overview image` with a maximum resolution of MAX_VIEW_PIXELS.
- You can call the **focus** tool to request detailed views for specific regions. Both overview and focused regions are constrained by MAX_VIEW_PIXELS.
You should perform the `focus` search until you are completely SURE that the question can be solved. 

# Tools
You are provided with the function signature within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "focus",
        "description": "Request a detailed view of the overview image from the original image pixel space. The returned focused image will still be contrained to MAX_VIEW_PIXELS if too large.",
        "parameters": {
            "type": "object",
            "properties": {
                "bboxes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "The bounding box of the region to crop, as [x1, y1, x2, y2] in ABSOLUTE PIXEL COORDINATES of the overview image."
                    },
                    "description": "A list of bounding boxes to zoom in on. You can request 1-3 bboxes at a turn."
                }
            },
            "required": ["bboxes"]
        }
    }
}
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
INSTRUCTION_PROMPT_SYSTEM_FIX_RETINA_v1="""You are a helpful assistant.

# Tools
You are provided with the function signature within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "focus",
        "description": "Request a high-resolution local region of the first image and zoom in",
        "parameters": {
            "type": "object",
            "properties": {
                "bboxes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "The bounding box of the region to crop, as [x1, y1, x2, y2] in ABSOLUTE PIXEL COORDINATES of the first image."
                    },
                    "description": "A list of bounding boxes to zoom in on. You can request 1-3 bboxes at a turn."
                }
            },
            "required": ["bboxes"]
        }
    }
}
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



# pixelreasoner
INSTRUCTION_PROMPT_SYSTEM_PIXELREASONER="""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image_normalized", "description": "Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. Values should be within [0.0,1.0].", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "select_frames", "description": "Select frames from a video.", "parameters": {"type": "object", "properties": {"target_frames": {"type": "array", "description": "List of frame indices to select from the video (no more than 8 frames in total).", "items": {"type": "integer", "description": "Frame index from 1 to 16."}}, "required": ["target_frames"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "crop_image_normalized", "arguments": {"bbox_2d": [0.1, 0.2, 0.3, 0.4], "target_image": 1}}  
</tool_call>

"""

# deepeyes
INSTRUCTION_PROMPT_SYSTEM_DEEPEYES="""You are a helpful assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>
"""

# visionthink
INSTRUCTION_PROMPT_SYSTEM_VISIONTHINK="""You are a helpful assistant.

# Tools

You may call the function tool shown below to assist with the user query.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "resize_image", "name": "resize_image", "description": "Resize the image resolution.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `resize`: Double the resolution of the current image. You should only use this tool if you are unable to obtain the critical information needed to answer the question from the current resolution.", "enum": ["resize"], "type": "string"}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>
For each function call, return a json object with the function name and the corresponding argument within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
# adaptvision
INSTRUCTION_PROMPT_SYSTEM_ADAPTVISION = """You are a helpful assistant.

# Tools
You may call the function tool shown below to assist with the user query.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "request_local_region", "name": "request_local_region", "description": "Request a high-resolution local region of the current image and zoom in", "parameters": {"properties": {"bbox_2d": {"type": "array", "items":{"type":"integer"}, "minItems":4, "maxItems":4, "description": "The bounding box of the region to crop, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the target region and (x2, y2) is the bottom-right corner of the target region. The bounding box should be in the absolute pixel coordinates of the current image."}}, "required": ["bbox_2d"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>
For each function call, return a json object with the function name and the corresponding argument within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""
# Chain of Focus
INSTRUCTION_PROMPT_SYSTEM_COF = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) and an optional object label.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
# Mini-o3
INSTRUCTION_PROMPT_SYSTEM_MINIO3="""You are a helpful assistant. Answer the user's question based on the image provided. Output your thinking process within the <think> and </think> tags. Whenever you find anything unclear, you can zoom in a specific region in the given image to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], \"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region that you want to zoom in, respectively (suppose the width and height of the image are 1.0), and 'source' refers to the image that you zoom in and could be either 'original_image' or 'observation_i'. Once the final answer is confirmed, put it within <answer> and </answer>."""


# User prompts
AFTER_PROMPT="Think first, call **image_zoom_in_tool** if needed, then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer> "
USER_PROMPT_TEMPLATE = "<image>\n{dataset_prompt}\n{AFTER_PROMPT}"
USER_PROMPT_TEMPLATE_DEEPEYES="""{dataset_prompt}
\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> 
"""
USER_PROMPT_TEMPLATE_QWEN="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags. Here is the question: {dataset_prompt}\n"
USER_PROMPT_TEMPLATE_FIX_RETINA="""Think first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question: {dataset_prompt}\n"""
USER_PROMPT_TEMPLATE_PIXELREASONER="""{dataset_prompt}\n\nGuidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For a video, we can look closer by `select_frames`. For an image, we can look closer by `crop_image`. Reason with the visual information step by step, and put your final answer within \\boxed{{}}."""
USER_PROMPT_TEMPLATE_VISIONTHINK="Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. You may call ONE function tool per step to help you better solve the problem. Place the function tool within <tool_call> and </tool_call> at the end of each step to perform a function call. You should continue your reasoning process based on the content returned by the function tool. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question: {dataset_prompt}"
USER_PROMPT_TEMPLATE_ADAPTVISION="Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. You may call ONE function tool per step to help you better solve the problem. Place the function tool within <tool_call> and </tool_call> at the end of each step to perform a function call. You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question:\n{dataset_prompt}"
USER_PROMPT_TEMPLATE_COF = "{dataset_prompt}\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
USER_PROMPT_TEMPLATE_MINIO3 = "{dataset_prompt} Please reason step by step. Output the thinking process in <think> </think> and put your final answer (number) within <answer> \\boxed{{}} </answer>."
TOOL_CALL_CROP_MULTI_TRUN_PROMPT_MINIO3="After the above Action {action_turn}, here is the the zoom-in image (Observation {observation_turn}):\n{observation_image}\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."
# Tool call tokens
TOOL_CALL_START_TOKEN = "<tool_call>"
TOOL_CALL_END_TOKEN = "</tool_call>"
ANSWER_START_TOKEN = "<answer>"
ANSWER_END_TOKEN = "</answer>"


