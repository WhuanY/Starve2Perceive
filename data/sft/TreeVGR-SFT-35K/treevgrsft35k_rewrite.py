import json
import json_repair
import re
import logging
from openai import OpenAI
from PIL import Image
import base64
import os
import traceback

SYSTEM_PROMPT="""You are a helpful assistant.

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
</tool_call>"""


from openai import OpenAI
from PIL import Image
import base64
import os
import json

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("treevgrsft35k_rewrite")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Best-effort file logging for post-mortem debugging.
    try:
        fh = logging.FileHandler("treevgrsft35k_rewrite.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # Don't crash if filesystem isn't writable
        pass

    return logger


logger = _setup_logger()

client = OpenAI(
    base_url="https://open.xiaojingai.com/v1", 
    api_key=os.environ.get("OPENAI_API_KEY")
)
rewrite_system_prompt = """You are an expert data synthesist for training visual agents.
Your task is to convert "Passive Grounding" data (descriptions with bounding boxes) into "Active Visual Search" training data AS IF the agent is equipped with \
the following tool:

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

**Input Format:**
You will receive a User Question and a GPT Response containing reasoning and bounding boxes like `<think>...<box>[x1, y1, x2, y2]</box></think><answer></answer>`. 

**Output Goal:**
Rewrite the GPT Response into a **Multi-step** format where the agent:
1. **Thinks** about the need to see the object clearly to answer the user.
2. **Decides** to call a `zoom` tool on the specific region provided in the input.
3. **Do Self-correction** on the wrongly focused image by analyze other cropped image or zoom in on another area.
If a bbox comes along with the meta prompt `Wait, this box seems wrong.`, then this bbox is regarded as a wrongly focused image.

**Strict Rules**
- PRESERVE the original bounding box coordinates exactly. DO NOT invent ANY new coordinates.
- ALL bboxes bounded by <box> ... </box> should be used in the output but ONLY between <tool_call> </tool_call>. 
- The format should follow <think>...</think> <tool_call>...</tool_call> <tool_response>...</tool_response> OR <think> ... </think> <answer> ... </answer> at each turn.
- The bbox number requested in <tool_call>...</tool_call> should match the number in the tool response and no more than 3 at each turn.
- The tokens `<box>` and `</box>` should NEVER APPEAR in the output.

**Output Format:**
Your output format should be STRICTLY json and be in a sharedgpt multi-turn(can be 3, 4, or 5 turns) format. 
```json
[
    {
        "from": "gpt",
        "value": "<think> ... </think> <tool_call> ... </tool_call>"
    },
    {
        "from": "human",
        "value": "<tool_response><image><tool_response>"
    },
    {
        "from": "gpt",
        "value": "<think> ... </think><answer> ...</answer>
    }
]
```
Here are a few examples for the input and output format:
"""

def get_ICE_examples():
    # TODO: Add ICE examples here
    ret_str = r"""
Example 1 (2 bboxes. No pertubation):
[USER QUESTION] Who is wearing glasses?
[GPT RESPONSE] <think>To determine who is wearing glasses in the image, there are two people present. The person on the left <box>[0,0,205,375]</box> has dark hair, is wearing a white-shirt under a dark vest, and eyeglasses are clearly visible on their face. The person on the right <box>[360,78,500,375]</box> has their hair tied back, is wearing an orange top, and no eyeglasses are visible on them. So the one wearing glasses is the person on the left, who is a woman.
</think><answer>Woman</answer>
[OUTPUT] ```json
[
    {
        "from": "gpt",
        "value": "<think>To determine who is wearing glasses, I need to examine the people in the image more closely. Let me zoom in on them.</think> <tool_call>{\"name\": \"focus\", \"arguments\": {\"bboxes\": [[0,0,205,375], [360,78,500,375]]}}</tool_call>"
    },
    {
        "from": "human",
        "value": "<tool_response><image><image></tool_response>"
    },
    {
        "from": "gpt",
        "value": "<think>Now I can see clearly. The person on the left has dark hair, is wearing a white shirt under a dark vest, and eyeglasses are clearly visible on their face. The person on the right has their hair tied back, is wearing an orange top, and no eyeglasses are visible. So the one wearing glasses is the person on the left, who is a woman.</think><answer>Woman</answer>"
    }
]
```

Example 2 (3 bboxes with perturbation.):
[USER QUESTION] Does the sheep on the grass appear to be white and large?
[GPT RESPONSE] <think>
The question asks if the sheep on the grass appear to be white and large. Looking at the image, there are multiple sheep in a fenced-in area on green grass. Now, let's look at <box>[263,316,485,370]</box>.
Wait, this box seems wrong. Let me check again.
Consider the sheep in the foreground <box>[0,97,300,375]</box>. Its wool is light-colored, close to white, and it has an adult-sized body, which is large. Also, the middle sheep <box>[145,90,420,345]</box> has a similar light-colored wool and a robust, large build. All the sheep in the image have thick, light-colored wool and adult-sized bodies, so they appear white and large.
</think>
<answer>Yes</answer>
[OUTPUT] ```json
[
    {
        "from": "gpt",
        "value": "<think>The question asks if the sheep on the grass appear to be white and large. I can see multiple sheep in a fenced-in area on green grass, but I need to examine them more closely to determine their color and size. Let me zoom in on some sheep.</think> <tool_call>{\"name\": \"focus\", \"arguments\": {\"bboxes\": [[263,316,485,370], [0,97,300,375], [145,90,420,345]]}}</tool_call>"
    },
    {
        "from": "human",
        "value": "<tool_response><image><image><image></tool_response>"
    },
    {
        "from": "gpt",
        "value": "<think>The first region doesn't show the sheep clearly. Let me focus on the other regions. Looking at the sheep in the foreground, its wool is light-colored, close to white, and it has an adult-sized body, which is large. Also, the middle sheep has a similar light-colored wool and a robust, large build. All the sheep in the image have thick, light-colored wool and adult-sized bodies, so they appear white and large.</think><answer>Yes</answer>"
    }
]
```

Example 3 (4 bboxes with perturbation):
[USER QUESTION] Is this a journey related book?
[GPT RESPONSE] <think>
The question asks if the book is journey-related. Analyzing the cover, the title "Yukon" <box>[62,0,235,160]</box> is a well-known region in Canada, often associated with travel and exploration due to its natural beauty and outdoor activities. Now, let's look at <box>[259,243,273,262]</box>.
Wait, this box seems wrong. Let me check again.
The book is part of "Bradt Guides" <box>[210,60,310,235]</box>, a publisher known for travel guides. The endorsement by Michael Palin, a well-known traveler, further supports this. The cover image of caribou in a natural landscape <box>[0,105,288,495]</box> is typical of what one would see in a travel guide about a place like Yukon. All these elements together indicate that the book is related to travel or a journey.
</think>
<answer>Yes</answer>
[OUTPUT] ```json
[
    {
        "from": "gpt",
        "value": "<think>The question asks if this book is journey-related. I see a small detail on the cover that might be a logo or a travel-related mark at [259, 243, 273, 262]. Let me zoom in to verify if this provides a direct clue.</think> <tool_call>{\"name\": \"focus\", \"arguments\": {\"bboxes\": [[259, 243, 273, 262]]}}</tool_call>"
    },
    {
        "from": "human",
        "value": "<tool_response><image></tool_response>"
    },
    {
        "from": "gpt",
        "value": "<think>Wait, the cropped region doesn't provide useful information; it appears to be a minor background detail. I need to pivot and examine the primary identifiers of the book, such as the title and the publisher's branding. Let me focus on the text areas at the top and bottom.</think> <tool_call>{\"name\": \"focus\", \"arguments\": {\"bboxes\": [[62, 0, 235, 160], [210, 60, 310, 235]]}}</tool_call>"
    },
    {
        "from": "human",
        "value": "<tool_response><image><image></tool_response>"
    },
    {
        "from": "gpt",
        "value": "<think>Now I have a clear view. The title \"Yukon\" refers to a well-known region in Canada associated with exploration. Additionally, the branding \"Bradt Guides\" confirms this is from a famous travel guide publisher. I should also check the main cover illustration to be completely sure.</think> <tool_call>{\"name\": \"focus\", \"arguments\": {\"bboxes\": [[0, 105, 288, 495]]}}</tool_call>"
    },
    {
        "from": "human",
        "value": "<tool_response><image></tool_response>"
    },
    {
        "from": "gpt",
        "value": "<think>The view of the cover shows caribou in a natural landscape, which is typical for travel literature about the Yukon. Combined with the Bradt Guides publisher and the endorsement by Michael Palin, it is clear this book is journey-related.</think><answer>Yes</answer>"
    }
]
```"""
    return ret_str.strip()


rewrite_system_prompt += get_ICE_examples()


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
        b64_image = base64.b64encode(image_data).decode('utf-8')
    return b64_image


def is_valid(original_bboxes: list[list[int]], rewrited_json: list[dict], *, sample_tag: str = "") -> bool:
    """
    Validate the rewritten JSON conversation format.
    
    Checks:
    1. At least 3 turns
    2. Alternating gpt/human/gpt pattern
    3. First message is from gpt, last message is from gpt with <answer>
    4. Each gpt message has <think>...</think>
    5. Tool call bbox count matches tool response image count.
    6. Proper format: <think>...</think> <tool_call>...</tool_call> or <think>...</think> <answer>...</answer>
    7. Each tool call has bboxes no more than 3.
    """
    def fail(reason: str) -> bool:
        tag = f" [{sample_tag}]" if sample_tag else ""
        logger.warning(f"is_valid failed{tag}: {reason}")
        return False

    # 1. Should be at least 3 turns
    if len(rewrited_json) < 3:
        return fail(f"len(rewrited_json)={len(rewrited_json)} < 3")
    
    # 2. First message should be from gpt, last should be from gpt with answer
    if rewrited_json[0].get('from') != 'gpt':
        return fail("first msg is not from gpt")
    if rewrited_json[-1].get('from') != 'gpt':
        return fail("last msg is not from gpt")
    if '<answer>' not in rewrited_json[-1].get('value', ''):
        return fail("last gpt msg missing <answer>")
    
    # 3. Validate alternating pattern and message formats
    for idx, msg in enumerate(rewrited_json):
        if 'from' not in msg or 'value' not in msg:
            return fail(f"msg[{idx}] missing 'from' or 'value'")
        
        role = msg['from']
        content = msg['value']
        
        if role == 'gpt':
            # Every gpt message must have <think>...</think>
            if '<think>' not in content or '</think>' not in content:
                return fail(f"msg[{idx}] gpt missing <think>...</think>")

            if ['<box>', '</box>'] in content:
                return fail(f"msg[{idx}] gpt has <box> or </box> in the content")
                
            # Check if this is a tool call message
            if '<tool_call>' in content:
                if '</tool_call>' not in content:
                    return fail(f"msg[{idx}] has <tool_call> without </tool_call>")
                
                # Extract bbox count from tool_call
                tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                if tool_call_match:
                    tool_call_json = tool_call_match.group(1)
                    # Parse the tool call to get bbox count
                    try:
                        # Allow slightly-invalid json
                        tool_call_data = json_repair.loads(tool_call_json)
                        if 'arguments' in tool_call_data and 'bboxes' in tool_call_data['arguments']:
                            bbox_count = len(tool_call_data['arguments']['bboxes'])
                            if bbox_count < 1 or bbox_count > 3:
                                return fail(f"msg[{idx}] tool_call bbox_count={bbox_count} not in [1,3]")
                            
                            # Next message should be human with matching image count
                            if idx + 1 < len(rewrited_json):
                                next_msg = rewrited_json[idx + 1]
                                if next_msg.get('from') != 'human':
                                    return fail(f"msg[{idx}] tool_call next msg is not human")
                                
                                # Count <image> tags in tool_response
                                tool_response_match = re.search(r'<tool_response>(.*?)</tool_response>', next_msg.get('value', ''), re.DOTALL)
                                if tool_response_match:
                                    tool_response_content = tool_response_match.group(1)
                                    image_count = tool_response_content.count('<image>')
                                    if image_count != bbox_count:
                                        return fail(f"msg[{idx}] bbox_count={bbox_count} != next human image_count={image_count}")
                                else:
                                    return fail(f"msg[{idx}] next human missing <tool_response>...</tool_response>")
                            else:
                                return fail(f"msg[{idx}] tool_call but no next human msg")
                        else:
                            return fail(f"msg[{idx}] tool_call missing arguments.bboxes")
                    except Exception as e:
                        logger.exception(f"is_valid exception{(' ['+sample_tag+']') if sample_tag else ''}: tool_call parse failed: {e}")
                        return False
            
            # Last gpt message must have <answer>
            if idx == len(rewrited_json) - 1:
                if '<answer>' not in content or '</answer>' not in content:
                    return fail("last gpt msg missing <answer>...</answer>")
        
        elif role == 'human':
            # Human messages must have <tool_response>...</tool_response>
            if '<tool_response>' not in content or '</tool_response>' not in content:
                return fail(f"msg[{idx}] human missing <tool_response>...</tool_response>")
            
            # Should have at least one <image> tag
            if '<image>' not in content:
                return fail(f"msg[{idx}] human tool_response missing <image>")
        else:
            return fail(f"msg[{idx}] unknown role={role}")
    
    # 4. Validate alternating pattern (gpt -> human -> gpt -> ...)
    for idx in range(len(rewrited_json) - 1):
        current_role = rewrited_json[idx]['from']
        next_role = rewrited_json[idx + 1]['from']
        if current_role == next_role:
            return fail(f"roles not alternating at idx={idx}: {current_role} -> {next_role}")

    # 5. Check if the bboxes are the same as the original bboxes
    expected_set = [tuple(b) for b in original_bboxes]
    actual_set = []
    for msg in rewrited_json:
        if msg['from'] == 'gpt' and '<tool_call>' in msg['value']:
            try:
                tc_match = re.search(r'<tool_call>(.*?)</tool_call>', msg['value'], re.DOTALL)
                if tc_match:
                    tc_data = json_repair.loads(tc_match.group(1))
                    bboxes = tc_data['arguments']['bboxes']
                    if len(bboxes) > 3 or len(bboxes) < 1:
                        return fail(f"tool_call bboxes len={len(bboxes)} not in [1,3]")
                    for b in bboxes:
                        actual_set.append(tuple(b))
            except Exception:
                logger.exception(f"is_valid exception{(' ['+sample_tag+']') if sample_tag else ''}: extracting bboxes from tool_call failed")
                return False

    if len(actual_set) != len(expected_set):
        return fail(f"bbox count mismatch: expected={len(expected_set)} actual={len(actual_set)}")
    
    if sorted(actual_set) != sorted(expected_set):
        return fail(f"bbox set mismatch: expected={sorted(expected_set)} actual={sorted(actual_set)}")

    return True
            


def rewrite_gpt(images: list[str], gpt_response: str, user_question: str) -> list[dict]:
    """
    Rewrite GPT response from passive grounding format to multi-turn active visual search format.
    
    Args:
        images: List of image paths
        gpt_response: Original GPT response string containing reasoning and bounding boxes
        user_question: User question string
    
    Returns:
        List of dictionaries containing rewritten conversation in multi-turn format
    """
    # Load and encode the first image
    if not images or len(images) == 0:
        raise ValueError("No images provided")
    
    image_path = images[0]
    base_dir = "/home/zzhengbo/vlmpaper/data/TreeVGR-SFT-35K/llava_next_raw_format"
    image_path = os.path.join(base_dir, image_path)
    try:
        b64_image = encode_image_to_base64(image_path)
    except Exception:
        logger.exception(f"rewrite_gpt failed: encode_image_to_base64 error. image_path={image_path}")
        return None
    
    # Prepare the input prompt for rewriting
    input_text = f"""[USER QUESTION] {user_question}
[GPT RESPONSE] {gpt_response}
"""
    
    # Call OpenAI API to rewrite
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {'role': "system", "content": rewrite_system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        {"type": "text", "text": input_text}
                    ]
                }
            ]
        )

        response_str = response.choices[0].message.content
        rewrited_json = parse_json_str(response_str)
    except Exception:
        logger.exception(
            "rewrite_gpt failed: API/parse error. "
            f"image_path={image_path} user_question={user_question[:120]!r}"
        )
        return None
        

    original_bboxes = extract_bboxes(gpt_response)
    sample_tag = f"img={images[0]} q={user_question[:60].replace(chr(10),' ')!r}"
    if is_valid(original_bboxes, rewrited_json, sample_tag=sample_tag):
        return rewrited_json
    else:
        logger.warning(
            "rewrite_gpt invalid output -> None. "
            f"{sample_tag} original_bbox_cnt={len(original_bboxes)}"
        )
        return None


def rewrite_conversations(images, conversations: list[dict]) -> list[dict]:
    user_question = conversations[0]['value'].replace("<image>\n", "").replace("<image>", "").strip()
    user_prompt="<image>\nThink first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question: \n"
    user_prompt += user_question
    new_conversation = [
        {"from": "human", "value": user_prompt},
    ]
    rewrite_gpt_response_sharegpt = rewrite_gpt(images, conversations[1]['value'], user_question)
    if rewrite_gpt_response_sharegpt is None:
        logger.warning(
            "rewrite_conversations -> None (rewrite_gpt returned None). "
            f"img={images[0] if images else None} q={user_question[:120]!r}"
        )
        return None
    # 批量append
    for i in range(len(rewrite_gpt_response_sharegpt)):
        new_conversation.append(rewrite_gpt_response_sharegpt[i])
    
    return new_conversation


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def parse_json_str(json_str: str) -> dict:
    return json_repair.loads(json_str)


def extract_bboxes(reply):
    "<think>\nThe question asks which of the given options feed on bacteria. Analyzing the diagram, there are arrows pointing from bacteria to <box>[168,0,238,94]</box>, <box>[191,67,250,133]</box>, and <box>[185,113,255,162]</box>. This indicates that all three-tube worms, white clams, and yellow mussels-consume bacteria.\n</think>\n<answer>D</answer>"
    "-> [[168,0,238,94], [191,67,250,133], [185,113,255,162]]"
    bboxes = []
    # use re to find all 
    bboxes = re.findall(r'<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>', reply)
    # convert each tuple to [int, int, int, int]
    bboxes = [[int(x) for x in bbox] for bbox in bboxes]
    return bboxes


def spilt_question_and_options(question_str: str) -> tuple[str, list[str]]:
    # 1. 清理题目主体
    # 去掉 <image>\n (如果有)
    q_text = re.sub(r'^<image>\s*', '', question_str)
    
    # 2. 提取选项
    # 寻找形如 "A. text" 或 "A: text" 的模式
    options = re.findall(r'[A-Z][\.\:]\s*(.+?)(?=\s*[A-Z][\.\:]|$)', q_text)
    
    # 3. 提取纯题目
    # 将第一个选项及其后面的内容全部切掉
    main_question = re.split(r'\s*[A-Z][\.\:]', q_text)[0].strip()
    
    options = [opt.strip() for opt in options]
    
    return main_question, options


def convert_single_record(record):
    """
    Convert a single record from passive grounding format to active visual search format.
    
    Args:
        record: Dictionary containing 'images', 'conversations', and 'system' keys
    
    Returns:
        Dictionary with converted data, or None if conversion fails
    """
    meta_cognitive_prompt = "Wait, this box seems wrong."
    output = {
        "images": record['images'],
        "raw_conversations": record['conversations'],
        "conversations": [],
        "answer": "",
        "options": [],
        "question": "",
        "perturbation": False,
        "system": record['system']
    }
    raw_gpt = record['conversations'][1]['value']
    assert record['conversations'][1]['from'] == 'gpt'
    raw_human = record['conversations'][0]['value']
    assert record['conversations'][0]['from'] == 'human'

    # ===answer===
    try:
        answer = raw_gpt.split('<answer>')[1].split('</answer>')[0].strip()
        output['answer'] = answer
    except:
        logger.warning(
            "convert_single_record -> None (answer parse failed). "
            f"img={record.get('images', [None])[0]} raw_gpt_prefix={raw_gpt[:200]!r}"
        )
        return None

    # ===questions, options===
    question = raw_human.split("<image>\n")[1]
    options = []
    if "?\nA." in question and "\nB." in question:
        question, options = spilt_question_and_options(question)
    
    if "Answer the question with GPT-T-COCO format." in question:
        question = question.replace("Answer the question with GPT-T-COCO format.", "")
    output['question'] = question
    output['options'] = options
    
    # ===perturbation===
    output['perturbation'] = meta_cognitive_prompt in raw_gpt
    
    # === conversations ===
    output['conversations'] = rewrite_conversations(record['images'], record['conversations'])
    if not output['conversations']:
        logger.warning(
            "convert_single_record -> None (rewrite_conversations returned None/empty). "
            f"img={record.get('images', [None])[0]} q={output.get('question','')[:120]!r}"
        )
        return None
    # === system ===
    output['system'] = SYSTEM_PROMPT

    return output


if __name__ == "__main__":
    from tqdm import tqdm
    data = read_json("TreeVGR-SFT-35K.json")
    rewrite_data = []
    d_with_meta_cognitive = []
    meta_cognitive_prompt = "Wait, this box seems wrong."
    for d in data:
        gpt = d['conversations'][1]['value']
        if meta_cognitive_prompt in gpt:
            d_with_meta_cognitive.append(d)
    print(len(d_with_meta_cognitive))
    sampled = d_with_meta_cognitive[:10]
    print(len(sampled))
    # print(sampled[0])
    for s in tqdm(sampled, total=len(sampled), desc="Converting data"):
        rewrited = convert_single_record(s)
        if rewrited:
            rewrite_data.append(rewrited)


    print(len(rewrite_data))
        

    temp_res_path = "temp_res.json"
    with open(temp_res_path, 'w') as f:
        json.dump(rewrite_data, f, indent=4)
    print(f"Saved to {temp_res_path}")