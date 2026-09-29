import os
import random
from typing import Tuple, List
import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", None)
]
client_list = []
for api_base in openai_api_base_list:
    if api_base is not None:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=api_base,
        )
        client_list.append(client)

model_name_list = [
    "judge"
] 

def _map_str_to_choice(options: list[str], pred: List[str] | str) -> str | None:
    """
    extract multiple choice answer from the prediction
    """
    import re
    if not pred:
        return ""
        
    assert len(options) > 0
    if isinstance(pred, list): # golden answer
        pred = pred[0]
    assert isinstance(pred, str)
    pred = str(pred).strip()
    
    # 1. Single character ("A")
    if len(pred) == 1 and pred.isalpha():
        return pred.upper()
        
    # 2. Patterns
    patterns = [
        r'(?i)answer\s*is\s*:?\s*\(?([A-Z])\)?', # "answer is: (A)"
        r'(?i)the\s*answer\s*is\s*\(?([A-Z])\)?', # "the answer is (A)"
        r'(?i)answer\s*:\s*\(?([A-Z])\)?', # "answer: (A)"
        r'(?i)choice\s*is\s*\(?([A-Z])\)?', # "choice is (A)"
        r'(?i)option\s*is\s*\(?([A-Z])\)?', # "option is (A)"
        r'(?i)correct\s*option\s*is\s*\(?([A-Z])\)?' # "correct option is (A)"
        # A: a woven blue mat
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, pred))
        if matches:
            return matches[-1].group(1).upper()
            
    # 3. Start patterns: 
    # start_patterns = [
        # r'^([A-Z])\.',  # A.
        # r'^\(([A-Z])\)',  # (A)
        # r'^([A-Z])\)',  # A)
        # r'^([A-Z]):'  # A:
    start_patterns = [
        r'^[\'"]?([A-Z])[\'"]?\.',  # A. 或 "A".
        r'^[\'"]?\(([A-Z])\)',      # (A)
        r'^[\'"]?([A-Z])\)',        # A)
        r'^[\'"]?([A-Z])[\'"]?:'    # A: 或 "A" 
    ]
    for p in start_patterns:
        m = re.search(p, pred)
        if m:
            return m.group(1).upper()
            
    # 4. End pattern: "(A)" at end
    m = re.search(r'\(([A-Z])\)\.?\s*$', pred)
    if m:
        return m.group(1).upper()

    # 5. pred string exact match with one of the options
    if pred in options:
        return chr(options.index(pred) + 65) # 65 is the ASCII code for 'A'

    return None

def _em(pred, answer):
    """
    Exact match
    """
    if isinstance(answer, list):
        return pred.strip().lower() in [a.strip().lower() for a in answer]
    elif isinstance(answer, str):
        return pred.strip().lower() == answer.strip().lower()
    else:
        raise ValueError(f"Unsupported answer type: {type(answer)}")

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    example_8 = """
[Question]: Is the cup on the left or right side of the wine glass?
[Standard Answer]: ['The cup is on the left side of the wine glass.']
[Model_answer] : 左</answer<th<think>思考><ththink<th><think>Now that I am confident</think><answer>左
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7, example_8]

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If [Model_answer] is uncommon compared to natural language (e.g., contains gibberish, non-natural symbols, repeated characters, or formatting errors), Judgement is 0.

If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

def extract_answer(chatml_histroy):
    last_assistant_msg = chatml_histroy[-1]
    if "assistant\n" not in last_assistant_msg:
        print(f"[WARNING] last message is not assistant message. {last_assistant_msg=}, {chatml_histroy=}")
        return None
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, last_assistant_msg, re.DOTALL))
    if len(matches) > 1:
        print(f"[WARNING] multiple <answer>...</answer> found in the last assistant message. {last_assistant_msg=}, {chatml_histroy=}")
        return None 
    if matches:
        return matches[-1].group(1).strip()
    
    return None



@retry(
    stop=stop_after_attempt(2), 
    wait=wait_exponential(multiplier=1, min=4, max=15)
)
def _lasj(pred, answer: List[str]| str, extra_info=None):
    """
    LLM as a Judge
    """
    if isinstance(answer, list):
        answer = str(answer)
        
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    
    # prompt = JUDGE_USER_PROMPT.format(question=extra_info['question'], ground_truth=answer, prediction=pred)
    prompt = get_prompt(pred, answer, extra_info['question'])
    completion = client.chat.completions.create(
        model=model_name_list[client_idx],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=5,
    )
    result = completion.choices[0].message.content.strip()

    with open("/map-vepfs/haozhe/yhwu/vlmpaper/tmp/check_lasj/qwen25_72bawq_judgehistory.jsonl", "a") as f:
        json.dump({
            "question": extra_info['question'],
            "options": extra_info.get('options', None),
            "ground_truth": answer,
            "prediction": pred,
            "result": result
        }, f)
        f.write('\n')  # 添加换行符，形成JSONL格式

    return bool("1" in result)


def _has_format_error(chatml_history):
    has_format_error = False
    for turn_idx, msg in enumerate(chatml_history):
        if turn_idx % 2 == 0: # 0, 2, ... assitant 
            msg = msg.lstrip("assistant\n")

            if (not msg.startswith("<think>")) or (msg.endswith("</think><|<im_end>|")):
                has_format_error = True
                break
            if msg.count("<think>") > 1 or msg.count("</think>") > 1:
                has_format_error = True
                break

            # ast_msg = .... </think> [<answer> ... </answer>] | [<tool_call> ... </tool_call>] <|im_end|>
            msg = msg.split("<think>")[-1] 

            msg_think, msg_after_think = msg.split("</think>")

            if msg_after_think.strip().startswith("<answer>"):
                if ("<tool_call>" in msg_think) or ("</tool_call>" in msg_think):
                    has_format_error = True
                    break
                if msg_after_think.count("<answer>") > 1 or msg_after_think.count("</answer>") > 1:
                    has_format_error = True
                    break
                if "<tool_call>" in msg_after_think or "</tool_call>" in msg_after_think:
                    has_format_error = True
                    break
                msg_after_think_after_answer = msg_after_think.split("<answer>")[-1]
                _ = msg_after_think_after_answer.replace("<|im_end|>", "")
                if (not _.rstrip().endswith("</answer>")):
                    has_format_error = True
                    break      
            elif msg_after_think.strip().startswith("<tool_call>"):
                if msg_after_think.count("<tool_call>") > 1 or msg_after_think.count("</tool_call>") > 1:
                    has_format_error = True
                    break
                if "<answer>" in msg_after_think or "</answer>" in msg_after_think:
                    has_format_error = True
                    break
                msg_after_think_after_tool_call = msg_after_think.split("<tool_call>")[-1]
                _ = msg_after_think_after_tool_call.replace("<|im_end|>", "")
                if _.strip().endswith("</tool_call>"):
                    continue
            
            else:
                print(f"[_has_format_error] wrong format: {msg_after_think=}")
                has_format_error = True

    return has_format_error

def is_correct(predict_str, ground_truth, extra_info=None):
    options= extra_info.get('options', None)
    if options is None:
        options = []
    is_mcq = len(options) > 0 and all(isinstance(option, str) for option in options)
    if is_mcq:
        pred_choice = _map_str_to_choice(options, predict_str.strip()) # Uppercase Single Char
        answ_choice = _map_str_to_choice(options, ground_truth) # Uppercase Single Char
        print(f"answ_choice: {answ_choice}")
        assert answ_choice is not None, f"answer choice is None. Something wrong with the datasource. Please check the data format"
        assert answ_choice in "ABCDEFGHIJKLMNOPQRSTUVWXYZ", f"answer choice is not a valid choice. Something wrong with the datasource. Please check the data format"
        if pred_choice and answ_choice: # 二者都能正确提取出来
            return pred_choice == answ_choice
    else: # free_form
        if len(client_list) == 0: # exact match
            return _em(predict_str, ground_truth)
        else:
            em_matched = _em(predict_str, ground_truth)
            if em_matched:
                print(f"[is_correct] em_matched: {predict_str=}\t{ground_truth=}")
                return True
    llm_judge_result = False
    try:
        llm_judge_result = _lasj(predict_str, ground_truth, extra_info)
    except:
        print(f"[WARNING] LLM Judge Failed. Return 0 instead!!! {predict_str=}\t{ground_truth=}")
    return llm_judge_result


def compute_score(predict_str, ground_truth, extra_info=None) -> float:
    tool_reward = 0.0
    correct = False
    chatml_histroy = predict_str.split("<|im_start|>")
    has_format_error = _has_format_error(chatml_histroy)
    answer_text = extract_answer(chatml_histroy)
    if answer_text is None:
        acc_reward = 0.0
        has_format_error = True
        correct = False
    else:
        if len(answer_text) < 200:
            correct = is_correct(answer_text, ground_truth, extra_info)  # only in this situation we judge the answer is correct
        else:  # avoid feeding too long answer to llm as judge. 
            acc_reward = 0.0
            has_format_error = True
            correct = False

    acc_reward = 1.0 if correct else 0.0
    format_reward = 0 if has_format_error else 1.0

    if ("<tool_call>" in predict_str) and ("</tool_call>" in predict_str) and ("\n<|im_start|>user\nError: " not in predict_str): 
        tool_reward = 1.0

    cond_tool_reward = acc_reward * tool_reward # conditional tool reward

    # debug
    if random.random() < 0.1:
        question = extra_info.get('question', None)
        options = extra_info.get('options', None)
        print(f"[compute_score] {question=}\t{options=}\t{predict_str=}\t{ground_truth=}\t{correct=}\t{has_format_error=}\t{tool_reward=}\t{acc_reward=}\t{format_reward=}")

    return {
        "score": acc_reward + format_reward + cond_tool_reward,
        "tool_reward": tool_reward,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
        "cond_tool_reward": cond_tool_reward
    }

def compute_score_w_reg(predict_str, ground_truth, extra_info=None) -> float:
    # print("[verl.utils.reward_score.fixretina.compute_score_w_reg] region reward is not implemented yet.")
    return compute_score(predict_str, ground_truth, extra_info) # TODO: 


if __name__ == "__main__":
#     predict_str = """<think>...
# </think>
# <tool_call>
# {"name": "focus", "arguments": {"bboxes": [[285, 105, 390, 197]]}}
# </tool_call>\n<|im_end|>
# <|im_start|>user\n<tool_response><|vision_start|><|image_pad|>...<|image_pad|><|vision_end|></tool_response><|im_end|>\n
# <|im_start|>assistant\n<think>..<tool_call>...</tool_call>.</think>\n<answer> A </answer>\n<|im_end|>"""
#     import pdb
    # sequences_str=['<think>\nThe image shows a scene at night with a pickup truck driving on a road. There is also a stop sign visible on the right side of the road. To determine which object is closer, I\'ll focus on the stop sign.\n\nI will use the `focus` tool to get a closer look at the stop sign.\n</think>\n<tool_call>\n{"name": "focus", "arguments": {"bboxes": [[315, 107, 385, 220]]}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|></tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nThe focused image shows the stop sign more clearly. Comparing the positions of the truck and the stop sign in the focused image, the stop sign is positioned higher up on the right side of the road, while the truck is positioned further down the road. Based on this, the stop sign is closer to the camera than the pickup truck.\n</think>\n<answer>B: The stop sign.</answer><|im_end|>', '<think>\nTo determine which object is closer, I need to analyze the image more closely. The pickup truck is visible on the right side of the image, and the stop sign is not clearly visible in this overview but might be near the road features.\n\nLet\'s use the `focus` tool to get a detailed view of the area near the truck and the stop sign to compare their positions.\n\nI\'ll focus on the area around the pickup truck and a potential location for the stop sign.\n</think>\n<tool_call>\n{"name": "focus", "arguments": {"bboxes": [[309, 109, 414, 217], [200, 147, 225, 170]]}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|></tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nFrom the focused images:\n\n1. The first focused image shows a close-up of the pickup truck, emphasizing its features such as the grill and part of the body. This provides a clear view of the truck\'s details.\n\n2. The second focused image shows a stop sign mounted on a pole. The stop sign is distinct and identifiable in the focused view.\n\nComparing the two focused images:\n- The pickup truck is clearly closer to the foreground in the first focused image.\n- The stop sign is not as prominently shown in detail but is still identifiable in the second focused image.\n\nGiven the options:\nA: The pickup truck.\nB: The stop sign.\n\nThe truck is positioned closer to the foreground and is more prominently featured in the images. Therefore, the pickup truck is closer.\n\n</think>\n<answer>A: The pickup truck.</answer><|im_end|>']
    # # pdb.set_trace()
    # for ss in sequences_str:
    #     __ = _has_format_error(ss)
    #     print(__)
    multi_turn_pred_str = """<think>\nThe image shows a person sitting behind a food stall and holding a bag of rice. There are no visible texts or symbols near the word "LIFE-" in the current overview image. To find the text following "LIFE-", I need to examine the area around the word "LIFE-" more closely.\n\nI will focus on the region containing the word "LIFE-" to see if there is any text present.\n</think>\n<tool_call>\n{"name": "focus", "arguments": {"bboxes": [[250, 197, 335, 243]]}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|></tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nThe focused image shows a close-up of the bag of rice but does not reveal any text. It appears that the focused area might not capture the correct region. Let me adjust the focus to a more precise area around the word "LIFE-" and its surroundings.\n</think>\n<tool_call>\n{"name": "focus", "arguments": {"bboxes": [[257, 197, 335, 222]]}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|></tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nThe focused image still does not show any text. It appears that the focused areas might not have sufficient detail to capture the text around "LIFE-". I will need to adjust my focus again to a different area, ensuring I cover the correct region for the word "LIFE-" and its surroundings.\n</think>\n<tool_call>\n{"name": "focus", "arguments": {"bboxes": [[270, 197, 350, 222]]}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|></tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nThe focused image still does not show any text. It seems that the focused areas are not capturing the correct region around the word "LIFE-". I will need to adjust my focus once more to a different area, ensuring I cover the correct region for the word "LIFE-" and its surroundings.\n</think>\n<tool_call>\n{"name": "focus", "arguments": {"bboxes": [[280, 197, 360, 222]]}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|></tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nAfter several attempts to focus on different areas around the word "LIFE-", it is evident that the focused regions do not reveal any text. Given the constraints and the fact that the word "LIFE-" is not fully visible or legible in the current images, I will provide an estimation based on the available information.\n\nThe word "LIFE-" is partially visible on the beige-colored bag of rice in the foreground. Based on the concentration of the bags and the overall context, it appears that the text following "LIFE-" might be related to the product details or branding. However, due to the resolution and focus constraints, I cannot confidently read or identify the text.\n\nGiven the constraints and the partial visibility of the text, I will estimate that the text following "LIFE-" could potentially be part of the brand name or product description on the bag, but the exact details are not clear.\n</think>\n<answer>The text following "LIFE-" is not clearly visible in the provided images. It appears to be part of the product label or branding on the bag of rice.</answer><|im_end|>"""
    # chatml_history=multi_turn_pred_str.split("<|im_start|>")
    # print(_has_format_error(chatml_history)) # why it return true
    print(compute_score(multi_turn_pred_str, ["Red"], extra_info={"question": "What is the text following \"LIFE-\", ", "options": [], "ground_truth": "Red"}))