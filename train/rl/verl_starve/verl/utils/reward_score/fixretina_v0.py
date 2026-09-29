import os
import random
from typing import Tuple, List
import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

JUDGE_SYSTEM_PROMPT=(
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs"
    "for question-answer pairs."
    "Your task is to compare the predicted answer with the correct answer and determine if they"
    "match meaningfully. Here’s how you can accomplish the task:\n"
    "INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)
JUDGE_USER_PROMPT=(
    "I will give you a question related to an image and the following text as inputs:\n"
    "1. **Question Related to the Image**: {question}\n"
    "2. **Ground Truth Answer**: {ground_truth}\n"
    "3. **Model Predicted Answer**: {prediction}\n"
    "Your task is to evaluate the model’s predicted answer against the ground truth answer, based "
    "on the context provided by the question related to the image. Consider the following criteria "
    "for evaluation:\n"
    "- **Relevance**: Does the predicted answer directly address the question posed, considering "
    "the information provided by the given question?\n"
    "- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to "
    "evaluate from the following two perspectives:\n"
    "(1) If the ground truth answer is open-ended, consider whether the prediction accurately "
    "reflects the information given in the ground truth without introducing factual inaccuracies. If "
    "it does, the prediction should be considered correct.\n"
    "The ground truth answer can also be a list. In this case, if ANY answer in the list matches"
    "the model's prediction, the prediction should be considered correct."
    "(2) If the ground truth answer is a definitive answer, strictly compare the model’s prediction "
    "to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long "
    "as the results are consistent, the model’s prediction should be deemed correct.\n"
    "**Output Format**:\n"
    "Your response should include an integer score indicating the correctness of the prediction: 1 "
    "for correct and 0 for incorrect. Note that 1 means the model’s prediction strictly aligns with "
    "the ground truth, while 0 means it does not.\n"
    "The format should be Score: 0 or 1\n"
)

JUDGE_TOOL_CALL_CONFIDENCE_SYSTEM_PROMPT = (
    "You are a Logic Auditor for a Visual Reasoning Agent. "
    "Your task is to analyze the agent's internal monologue (<think> content) "
    "and determine if the agent **ALREADY** believes it has sufficient visual evidence "
    "to answer the question directly, WITHOUT needing further verification."
)

JUDGE_TOOL_CALL_CONFIDENCE_PROMPT = (
    "I will provide a user question and the agent's thinking content.\n"
    "**Context**: The agent decided to call a 'zoom/focus' tool after this thought.\n"
    "**Task**: Determine if this tool call was REDUNDANT based on the thinking content.\n\n"
    
    "Input:\n"
    "1. **Question**: {question}\n"
    "2. **Thinking Content**: {think_content}\n\n"
    
    "**Classification Criteria**:\n"
    "- **CONFIDENT (Score 1)**: The agent explicitly states that the visual information is CLEAR, VISIBLE, or SUFFICIENT. The agent has already formed a firm conclusion based on the current view. (e.g., 'I can clearly see the text is...', 'The color is obviously red', 'No zoom is needed').\n"
    "- **UNCERTAIN (Score 0)**: The agent expresses doubt, ambiguity, blurriness, or a hypothesis that needs verification. The visual evidence is insufficient. (e.g., 'It looks like... but I need to check', 'The text is too small', 'I cannot be sure').\n\n"
    
    "**Goal**: We want to identify cases where the agent was CONFIDENT (1) but still wasted resources calling the tool.\n\n"
    
    "**Output Format**:\n"
    "Return ONLY the integer score.\n"
    "Score: <0 or 1>"
)

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

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.

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

def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
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
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=5,
    )
    result = completion.choices[0].message.content.strip()
    return bool("1" in result)


def _has_format_error(predict_str):
    predict_str = predict_str.strip()
    if not predict_str.startswith("<think>"):
        return True

    if predict_str.count("<think>") != predict_str.count("</think>"):
        return True

    if predict_str.count("<tool_call>") != predict_str.count("</tool_call>"):
        return True
     
    if "\n<|im_start|>user\nError: " in predict_str:
        return True

    if predict_str.count("<answer>") != predict_str.count("</answer>"):
        return True

    cleaned_str = predict_str.rstrip().replace("<|im_end|>", "").rstrip()
    if not cleaned_str.rstrip().endswith("</answer>"):
        return True

    return False

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
    has_format_error = _has_format_error(predict_str)
    answer_text = extract_answer(predict_str)
    if answer_text is None:
        answer_text = predict_str
        if len(answer_text) >= 200: 
            acc_reward = 0.0
            has_format_error = True
            correct = False
    else:
        if len(answer_text) < 200:
            correct = is_correct(answer_text, ground_truth, extra_info) 
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
    # predition_str = "<think>\nTo determine the color of the woven mat on which the child and woman are sitting, let's analyze the image:\n\n1. **Overview Analysis**: The mat appears to be blue in color.\n2. **Color Identification**: The mat has a fabric texture typical of woven materials, and its primary color is blue.\n\nGiven these observations, the most accurate description of the mat's color is blue.\n\nI am confident in my assessment without needing to use the `focus` tool, as the color is clearly visible in the overview image.\n\n</think>\n<answer>C: a woven blue mat</answer>"
    # prediction_str=r"""<think>The image shows a woman and a man in outdoor attire near a body of water, seemingly enjoying themselves. I see a detail regarding the color of their pants, but it seems a bit unclear. Let me focus on the pants to determine their color accurately.</think> <tool_call>{"name": "focus", "arguments": {"bboxes": [[143, 314, 445, 380], [213, 93, 332, 365]]}}</tool_call>\nuser\n<tool_response>Think first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer></tool_response>\nassistant\n<think>The upper-focused image does not show the woman\'s pants clearly, but it seems the focus was on the wrong area. Let\'s analyze the image carefully. The woman is holding a bottle and standing next to the man. Her short, red dress adds some contrast, but her pants are prominently visible. They appear to be a light, first-brown color. Comparing this to the other options, "fawn pants" best matches its hue.</think><answer>E</answer>"""
    # ground_truth = ['A']
    # extra_info={
    #     "options": ["fawn pants", "black pants", "brown pants", "green pants"]
    # }
    # print(compute_score(prediction_str, ground_truth, extra_info))

    # print("TEST 4")
    # print(_map_str_to_choice(["E"], """\n"E: a short-sleeved light-colored pink shirt"\n""".strip()))
    prediction_str = """<think>The question asks about the color of the blood vessels depicted in the image. I need to focus on the relevant area to discern the colors of the vessels. Let me zoom in on the coronary vessels area.</think> \n<tool_call>{"name": "focus", "arguments": {"bboxes": [[92, 71, 306, 445]]}}\n</tool_call>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|></tool_response><|im_end|>\n<|im_start|>assistant\n<think>Based on the first zoomed-in view, I can see that the blood vessels in this region are depicted in two colors: dark red (likely the arteries) and light blue (likely the veins). This matches option C: the red and blue colored blood vessels.</think><answer>C</answer><|im_end|>"""
    print(_has_format_error(prediction_str))
