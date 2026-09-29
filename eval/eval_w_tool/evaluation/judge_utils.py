import json
import os
from typing import Tuple
from PIL import Image
from openai import OpenAI
import random
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TOOL_CALL_CONFIDENCE_SYSTEM_PROMPT,
    JUDGE_TOOL_CALL_CONFIDENCE_PROMPT,
    get_prompt as get_lasj_ice_prompt,
)

def _IoU(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate the IoU of two bboxes.
    """
    x1_int = max(bbox1[0], bbox2[0])
    y1_int = max(bbox1[1], bbox2[1])
    x2_int = min(bbox1[2], bbox2[2])
    y2_int = min(bbox1[3], bbox2[3])

    if x2_int <= x1_int or y2_int <= y1_int:
        intersection_area = 0
    else:
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        iou = 0.0
    else:
        iou = intersection_area / union_area

    return iou      

class JudgeUtils:
    def __init__(self, base_url=None, openai_api_key=None, judge_model_name=None):
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.judge_model_name = judge_model_name
    
        if self.base_url is not None:
            self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        else:
            self.client = None

    def is_correct(self, sample: dict, pred: str, answer: str) -> bool:
        """Judge the correctness of the prediction."""
        if self._is_mcq(sample):
            # print("[is_correct] pred:", pred, "answer:", answer)
            pred_choice = self._map_str_to_choice(sample, pred) # Uppercase Single Char
            # print("[is_correct] pred_choice:", pred_choice)
            answ_choice = self._map_str_to_choice(sample, answer) # Uppercase Single Char
            # print("[is_correct] answ_choice:", answ_choice)
            return pred_choice == answ_choice

        else: # free_form
            if self.base_url is None: # exact match
                return self._em(pred, answer)
            else:
                try:
                    return self._lasj(sample, pred, answer)
                except Exception as e:
                    print(f"error in _lasj: {e}, sample: {sample}, pred: {pred}, answer: {answer}")
                    import traceback
                    traceback.print_exc()
                    return False

    def is_good_trajectory(self, sample: dict, traj_idx:int) -> (bool, str): 
        """
        Judge if a trajectory from a sample is good
        """
        question = sample['question']
        correct_lst = sample['correct']
        is_correct_sample = bool(correct_lst[traj_idx] == 1)
        enable_tool_call_confidence_filter = False # 先写死，如果不需要就不开了
        enable_bbox_quality_filter = True # 先写死，如果不需要就不开了
        if not is_correct_sample:
            return False, "non_correct_sample"

        pred_output = sample['pred_output'][traj_idx] # The chat history of the trajectory
        assitant_response = [msg['content'] for msg in pred_output if msg['role'] == 'assistant']

        right_format = True
        used_focus = False
        for ast_msg in assitant_response:
            assert isinstance(ast_msg, str), f"assistant response must be a string, got {type(ast_msg)}"
            # format check: <think> ... </think> ... [<answer> ... </answer>] | [<tool_call> ... </tool_call>]
            # must start with <think>
            ast_msg = str(ast_msg).strip()
            if (not ast_msg.startswith("<think>")) or (ast_msg.endswith("</think>")):
                right_format = False
                break

            # now that the ast msg starts with <think>, check if there is </think>
            ast_msg = ast_msg.split("<think>")[-1] # ast_msg = .... </think> [<answer> ... </answer>] | [<tool_call> ... </tool_call>]
            if "</think>" not in ast_msg:
                right_format = False
                break

            think_content, ast_msg_after_think = ast_msg.split("</think>") 

            ast_msg_after_think = ast_msg_after_think.strip()
            
            if ast_msg_after_think.startswith("<answer>") and ast_msg_after_think.endswith("</answer>"):
                continue

            elif ast_msg_after_think.startswith("<tool_call>") and ast_msg_after_think.endswith("</tool_call>"):
                used_focus = True
                if enable_tool_call_confidence_filter:
                    if self._llm_thinks_confident_for_answer(question, think_content):
                        print(f"tool call confidence filter. {think_content=}")
                        # right_format = False, "tool call confidence filter"
                        return False, "tool call confidence filter"
            else:
                right_format = False

        if not right_format:
            return False, "wrong_format"

        if enable_bbox_quality_filter:
            with Image.open(sample['image_path']) as img:
                img_size = img.size
            bboxes_list = sample['traj_info'][traj_idx]['bboxes_list']
            for bboxes in bboxes_list:
                # pdb.set_trace()
                if self._is_bad_bboxes(bboxes, sample, img_size):
                    return False, "bad_bboxes"

        # comparing with other correct trajectories within the group
        correct_traj_idxs = [i for i, c in enumerate(correct_lst) if c == 1]
        if len(correct_traj_idxs) == 1:
            # We found that this is the only correct trajectory among groups.
            return True, "only_correct_trajectory"
        
        # more than one correct trajectory within the group
        elif len(correct_traj_idxs) > 1:
            for correct_traj_idx in correct_traj_idxs:
                ast_response_len = len([msg['content'] for msg in sample['pred_output'][correct_traj_idx] if msg['role'] == 'assistant'])
                if ast_response_len < 2 and correct_traj_idx != traj_idx: # other trajectory answers question correctly without used focus.
                    return bool(not used_focus), "other correct trajectory no focus"
        else:
            assert False, "this should not happen"

        return True, "good_trajectory"

    def _is_mcq(self, sample: dict) -> bool:
        if sample['question'] == "Recognize the question and options in the image and answer it.": # treeBench
            return True
        options = sample.get("options", [])
        if options is None: # quick fix
            return False
        return len(options) > 0 and all(isinstance(option, str) for option in options)
    
    def _map_str_to_choice(self, sample, pred: str) -> str:
        """
        extract multiple choice answer from the prediction
        """
        import re
        if not pred:
            return ""
            
        pred = str(pred).strip()
        
        # 1. Single character ("A")
        if len(pred) == 1 and pred.isalpha():
            return pred.upper()
            
        # 2. Patterns (check last occurrence) ("answer is: (A)", "the answer is (A)", "answer: (A)", "choice is (A)", "option is (A)", "correct option is (A)")
        patterns = [
            r'(?i)answer\s*is\s*:?\s*\(?([A-Z])\)?',
            r'(?i)the\s*answer\s*is\s*\(?([A-Z])\)?',
            r'(?i)answer\s*:\s*\(?([A-Z])\)?',
            r'(?i)choice\s*is\s*\(?([A-Z])\)?',
            r'(?i)option\s*is\s*\(?([A-Z])\)?',
            r'(?i)correct\s*option\s*is\s*\(?([A-Z])\)?'
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, pred))
            if matches:
                return matches[-1].group(1).upper()
                
        # 3. Start patterns: "A.", "(A)", "A)"
        start_patterns = [r'^([A-Z])\.', r'^\(([A-Z])\)', r'^([A-Z])\)']
        for p in start_patterns:
            m = re.search(p, pred)
            if m:
                return m.group(1).upper()
                
        # 4. End pattern: "(A)" at end
        m = re.search(r'\(([A-Z])\)\.?\s*$', pred)
        if m:
            return m.group(1).upper()

        # 5. pred string exact match with one of the options
        options = sample.get("options", [])
        if pred in options:
            return chr(options.index(pred) + 65) # 65 is the ASCII code for 'A'

        return pred

    def _em(self, pred: str, answer: str|list[str]) -> bool:
        if isinstance(answer, list):
            return pred.strip().lower() in [a.strip().lower() for a in answer]
        elif isinstance(answer, str):
            return pred.strip().lower() == answer.strip().lower()
        else:
            raise ValueError(f"Unsupported answer type: {type(answer)}")


    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _lasj(self, sample: dict, pred: str, answer: str) -> bool:
        assert self.judge_model_name is not None, "judge_model_name is not set"
        # Use ICE-style prompt from fixretina (same as RL reward compute_score)
        prompt = get_lasj_ice_prompt(pred, answer, sample['question'])
        completion = self.client.chat.completions.create(
            model=self.judge_model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0,
            timeout=60,
        )
        
        # 提取结果
        result = completion.choices[0].message.content.strip()
        # 随机打印几个question, pred, answer, 写到文件里
        if random.random() < 0.1:
            with open("/map-vepfs/haozhe/yhwu/vlmpaper/tmp/check_lasj/lasj_ice_prompt_debug.txt", "a") as f:
                f.write(f"question: {sample['question']}, pred: {pred}, answer: {answer}, result: {result}\n")
        return bool("1" in result)

    def _llm_thinks_confident_for_answer(self, question: str, think_content: str) -> bool:
        prompt = JUDGE_TOOL_CALL_CONFIDENCE_PROMPT.format(question=question, think_content=think_content)
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_TOOL_CALL_CONFIDENCE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0,
            timeout=60,
        )
        result = completion.choices[0].message.content.strip()
        return bool("1" in result)

    def _is_bad_bboxes(self, bboxes: list[list[int]], sample:dict, img_size: Tuple[int, int]) -> bool:
        bboxes_num = len(bboxes)
        # too many bboxes
        if bboxes_num > 3:
            return True 

        # too much overlap
        iou_threshold = 0.6
        if bboxes_num > 1:
            for i in range(bboxes_num):
                for j in range(i + 1, bboxes_num):
                    if _IoU(bboxes[i], bboxes[j]) > iou_threshold:
                        print(f"iou: {_IoU(bboxes[i], bboxes[j])}")
                        return True

        area_max_threshold = 0.5
        # 1. bbox too large
        scale_x, scale_y = sample['traj_info'][0]['scale'] # each trajectory has the same scale
        orig_w, orig_h = img_size
        for bbox in bboxes:
            selected_x1, selected_y1, selected_x2, selected_y2 = bbox
            # mapping back x1, y1, x2, y2 to original pixel space
            raw_x1 = int(round(selected_x1 / scale_x))
            raw_y1 = int(round(selected_y1 / scale_y))
            raw_x2 = int(round(selected_x2 / scale_x))
            raw_y2 = int(round(selected_y2 / scale_y))

            if (raw_x2 - raw_x1) * (raw_y2 - raw_y1) > area_max_threshold * orig_w * orig_h:
                # 选中了太大的bbox
                return True

            # 2. bbox too small: 
            MIN_PIXELS = 4 * 28 * 28
            if (raw_x2 - raw_x1) * (raw_y2 - raw_y1) < MIN_PIXELS:
                return True

        # 3. TODO: more filters.
        return False

if __name__ == "__main__":
    judge_utils = JudgeUtils(base_url="https://open.xiaojingai.com/v1", openai_api_key=os.environ.get("OPENAI_API_KEY"), judge_model_name="gpt-4o-mini")
    # test_path = "/home/ywuit/vlmpaper/data/eval_w_tool_result/Qwen2.5-VL-32B-Instruct/fixretina_sft/Qwen2.5-VL-32B-Instruct/fixretina_sft/fixretina_sft_Qwen2.5-VL-32B-Instruct_n5_temperature1.0.jsonl"
    # test_path = "/home/ywuit/vlmpaper/eval/eval_w_tool/temp_v3.jsonl"
    test_path = "/home/ywuit/vlmpaper/tmp/100.jsonl"

    data = []
    with open(test_path, "r") as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
    
    for sample in data:
        # if sample['question'] == "What position is the car's front wheel relative to the woman's legs?":
        if sample['question']:
            print("question: ", sample["question"])
            print("pred_ans: ", sample['pred_ans'])

            print("is_mcq: ", judge_utils._is_mcq(sample))
            print("answer: ", sample["answer"])
            is_correct_lst = []
            for idx, pred in enumerate(sample['pred_ans']):
                # is_correct_lst.append(judge_utils.is_correct(sample, sample["answer"], pred))
                is_correct_lst.append(1) # 测试
            sample['correct'] = is_correct_lst
            sample['good_traj'] = []
            for idx, pred in enumerate(sample['pred_ans']):
                is_good_traj, reason = judge_utils.is_good_trajectory(sample, idx)
                print(f"is_good_trajectory for traj {idx}: {is_good_traj}, reason: {reason}")
                sample['good_traj'].append((is_good_traj, reason))


    # save temp sample to a new jsonl file 
    with open("/home/ywuit/vlmpaper/tmp/100_judged.jsonl", "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")
    # correct_list = [0,1,1,1]
    # sample['correct'] = correct_list
    # for idx, pred in enumerate(sample['pred_ans']):
    #     print(f"is_correct for pred {idx}: ", int(judge_utils.is_correct(sample, sample["answer"], pred)))
    # print("is_correct: ", judge_utils.is_correct(sample, sample["answer"], sample["pred_ans"][0]))
    # traj_idx = [1,2,3]
    # for idx in traj_idx:
    #     print(f"is_good_trajectory for traj {idx}: ", int(judge_utils.is_good_trajectory(sample, idx)))
    # print("is_good_trajectory: ", int(judge_utils.is_good_trajectory(sample, 1)))
    # test iou
    # boxA = [10, 10, 50, 50] # [x1, y1, x2, y2]
    # boxB = [30, 30, 70, 70]
    # iou = _IoU(boxA, boxB)
    # print("iou: ", iou)
       
