#!/usr/bin/env python3
"""
Main entry point for evaluating inference results.
Supports Best-of-N (BoN) evaluation and trajectory quality filtering.
"""
import argparse
import ast
import re
import sys
from pprint import pprint
import os
import json
import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation import JudgeUtils

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _calculate_average_iou(pred_boxes: List[List[float]], target_boxes: List[List[float]]) -> float:
    """Average IoU: for each target box, find best-matching pred box, then average."""
    if len(target_boxes) == 0:
        return 0.0

    total_iou = 0.0
    for t_coord in target_boxes:
        best_iou = 0.0
        for p_coord in pred_boxes:
            iou = _compute_iou(t_coord, p_coord)
            if iou > best_iou:
                best_iou = iou
        total_iou += best_iou

    return total_iou / len(target_boxes)


def extract_boxes_from_text(predict_str: str) -> List[List[int]]:
    """Extract boxes from <box>...</box> tags in text. Format: [x1,y1,x2,y2]."""
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, predict_str, re.DOTALL)

    all_boxes = []
    coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'

    for match in matches:
        box = match.strip()
        coord_match = re.match(coord_pattern, box)
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            if x1 < x2 and y1 < y2:
                all_boxes.append([x1, y1, x2, y2])

    return all_boxes


def extract_boxes_from_bboxes_list(
    bboxes_list: Tuple[List[List[List[int]]], List[List[int]]],
    scale: Union[List[float], tuple],
) -> List[List[float]]:
    """
    Extract and map bboxes from fixretina traj_info to original image coordinates.
    bboxes_list[i][j] = j-th bbox at i-th tool call turn (in overview/resized coords).
    scale = [scale_x, scale_y] maps overview -> original: orig = overview / scale.
    """
    if not scale or len(scale) < 2:
        scale_x, scale_y = 1.0, 1.0
    else:
        scale_x, scale_y = float(scale[0]), float(scale[1])

    all_boxes = []
    for turn_bboxes in bboxes_list:
        if isinstance(turn_bboxes, list) and isinstance(turn_bboxes[0], (float, int)): 
            turn_bboxes = [turn_bboxes]
        for bbox in turn_bboxes:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                # Map overview coords to original: orig = overview / scale
                orig_x1 = x1 / scale_x
                orig_y1 = y1 / scale_y
                orig_x2 = x2 / scale_x
                orig_y2 = y2 / scale_y
                if orig_x1 < orig_x2 and orig_y1 < orig_y2:
                    all_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])

    return all_boxes


def compute_box_iou(
    predict_str: Optional[str],
    target_boxes: List,
    record: Optional[Dict] = None,
    traj_idx: int = 0,
) -> float:
    """
    Compute average IoU between predicted and target boxes.
    Predicted boxes from: (1) <box> tags in predict_str, or (2) bboxes_list in traj_info.
    target_boxes: list of [x1, y1, x2, y2] in original image coordinates.
    """
    # Parse target_boxes if needed
    if target_boxes is None:
        target_boxes = []
    if isinstance(target_boxes, str):
        try:
            target_boxes = ast.literal_eval(target_boxes)
        except (ValueError, SyntaxError):
            target_boxes = []
    target_boxes = [[float(x) for x in b] for b in target_boxes if len(b) >= 4]

    pred_boxes = []

    # Source 1: <box> tags in text
    if predict_str:
        pred_boxes = extract_boxes_from_text(predict_str)

    # Source 2: bboxes_list from fixretina traj_info (when no <box> tags)
    if len(pred_boxes) == 0 and record:
        traj_infos = record.get('traj_info', [])
        if traj_idx < len(traj_infos):
            ti = traj_infos[traj_idx]
            bboxes_list = ti.get('bboxes_list', [])
            scale = ti.get('scale', [1.0, 1.0])
            pred_boxes = extract_boxes_from_bboxes_list(bboxes_list, scale)

    return _calculate_average_iou(pred_boxes, target_boxes)


def recursive_sum(l: List[Union[int, list]]) -> int:
    """
    Recursively sums up all integers in a nested list structure.

    Examples:
    [1, 2, 3, 4, 5] -> 15
    [[1, 2], [3, 4], [5]] -> 15
    [[[1,2], [3, 4]], [5]] -> 15
    """ 
    total = 0
    for item in l:
        if isinstance(item, list):
            total += recursive_sum(item)  # Recursively sum nested lists
        elif isinstance(item, int):
            total += item  # Add integers directly
    return total

def count_visual_tokens(results: List[dict[str, Any]], n: int ):
    """
    results: the loaded jsonl file for run_evaluation
    This script is used to count the average visual token consumption
    n: number of predictions to consider (matches args.n for Pass@N evaluation)
    """
    assert len(results) > 0, "No results found"
    assert n > 0, "n must be greater than 0"
    successful_traj_cnts = 0
    visual_token_cnts = 0
    
    for i,record in tqdm(enumerate(results), total=len(results)):
        overview_tokens = 0
        cropped_tokens = 0
        # Only consider the first n trajectories
        first_n_idxs = list(range(min(n, len(record['status']))))
        # Among the first n, find successful ones
        success_idxs = [idx for idx in first_n_idxs if record['status'][idx] == 'success']
        successful_traj_cnts += len(success_idxs)
        if len(success_idxs) > 0:
            for success_idx in success_idxs:
                overview_tokens += record['traj_info'][success_idx]['perf_stats']['overview_tokens']
                cropped_tokens += recursive_sum(record['traj_info'][success_idx]['perf_stats']['crop_tokens'])
        visual_token_cnts += overview_tokens + cropped_tokens

    visual_token_per_sample = visual_token_cnts / successful_traj_cnts if successful_traj_cnts > 0 else 0.0
    # print(f"Average visual token consumption per sample: {visual_token_per_sample}")
    return visual_token_per_sample

def calculate_mean_iou(results: List[Dict[str, Any]], n: int):
    """
    Calculate the mean IoU for the first n trajectories
    """
    assert len(results) > 0, "No results found"
    assert n > 0, "n must be greater than 0"
    iou_list = []
    for i,record in enumerate(results):
        # Only consider the first n trajectories
        first_n_idxs = list(range(min(n, len(record['status']))))
        # Among the first n, find successful ones
        success_idxs = [idx for idx in first_n_idxs if record['status'][idx] == 'success']
        if len(success_idxs) > 0:
            for success_idx in success_idxs:
                traj_info = record['traj_info'][success_idx]
                pred_flatten_bboxes_mapped_to_orig = extract_boxes_from_bboxes_list(
                    bboxes_list=traj_info['bboxes_list'],
                    scale=traj_info['scale']
                )
                if len(pred_flatten_bboxes_mapped_to_orig) > 0:
                    iou_list.append(record['iou'][success_idx])
    if len(iou_list) == 0:
        return 0.0
    mean_iou = np.mean(iou_list)
    return mean_iou

def eval_all_records(results: List[Dict[str, Any]], judge_utils: JudgeUtils, args):
    result_dict = {
        "#query": len(results),
        f"Pass@{args.n}": 0.0,
        f"Avg@{args.n}": 0.0,
        "Visual_tokens_per_sample": -1,
    }
    bon_correct_count = 0
    total_questions = len(results)

    def _eval_record(record: Dict[str, Any], args) -> Dict[str, Any]:
        answer = record.get('answer', '')
        pred_ans = record.get('pred_ans', [])
        pred_output = record.get('pred_output', [])
        if not isinstance(pred_ans, list):
            pred_ans = [pred_ans]

        # correct_list: List[int] = []
        correct_list: List[int] = record.get('correct', [])

        # first pass for checking correctness
        if not correct_list: # avoid duplicate checking
            for idx in range(len(pred_ans)):
                is_correct = judge_utils.is_correct(record, pred_ans[idx], answer)  # NOTE: Main
                correct_list.append(1 if is_correct else 0)
            record['correct'] = correct_list
        
        # pass@k calculation
        passk_inc = 0
        avgk_inc = 0
        num_predictions_to_check = args.n
        assert num_predictions_to_check > 0 and isinstance(num_predictions_to_check, int)
        first_n_predictions = correct_list[:num_predictions_to_check]
        has_any_correct = any(x == 1 for x in first_n_predictions)
        passk_inc = 1 if has_any_correct else 0
        avgk_inc = sum(first_n_predictions)
        
        # second pass for filtering out good trajectories
        if args.filter_good_traj:
            record['good_traj'] = []
            for idx in range(len(pred_ans)):
                is_good_traj = False
                reason = ""
                try:
                    is_good_traj, reason = judge_utils.is_good_trajectory(record, idx)  # NOTE: Main
                except Exception as e:
                    is_good_traj = False
                    reason = "error"
                    kill
                record['good_traj'].append((is_good_traj, reason))

        return {"record": record, "passk_inc": passk_inc, "avgk_inc": avgk_inc}

    updated_records: List[Dict[str, Any]] = [None] * total_questions
    passk_total_count: int = 0
    avgk_total_count: int = 0

    batch_size = getattr(args, 'batch_size', 20)
    num_batches = (total_questions + batch_size - 1) // batch_size

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_questions)
            batch_data = [(start_idx + i, results[start_idx + i]) for i in range(end_idx - start_idx)]

            future_to_idx = {
                executor.submit(_eval_record, rec, args): idx
                for idx, rec in batch_data
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                updated_records[idx] = res["record"]
                passk_total_count += res["passk_inc"]
                avgk_total_count += res["avgk_inc"]
    results = updated_records
   
    # accuracy
    passk_accuracy = passk_total_count / total_questions if total_questions > 0 else 0.0
    avgk_accuracy = avgk_total_count / (total_questions * args.n) if total_questions > 0 else 0.0

    # visual token counts
    if args.count_visual_tokens:
        visual_tokens_per_sample = count_visual_tokens(results, args.n)
        result_dict['Visual_tokens_per_sample'] = visual_tokens_per_sample

    result_dict[f'Pass@{args.n}'] = passk_accuracy
    result_dict[f'Avg@{args.n}'] = avgk_accuracy

    return results, result_dict
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA inference results")
    
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to inference results JSONL file')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save evaluation results. If not provided, overwrites input.')
    parser.add_argument('--dataset_name', type=str, default='HRBench4K',
                       help='Dataset name (used for evaluator initialization)')
    parser.add_argument('--n', type=int, default=1,
                       help='N for Pass@K and AVG@K evaluation.')
    parser.add_argument('--filter_good_traj', action='store_true',
                       help='Enable high-quality trajectory filtering. Adds "good_traj" list to output.')
    parser.add_argument('--count_visual_tokens', action='store_true',
                       help='Count the average visual token consumption per sample.')
    parser.add_argument('--base_url', type=str, default=None, 
                       help='Base URL for LLM judge (e.g. http://localhost:8000/v1)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                       help='API Key for LLM judge')
    parser.add_argument('--judge_model_name', type=str, default='gpt-4o-mini',
                       help='Model name for LLM judge')
    parser.add_argument('--max_workers', type=int, default=20,
                       help='Max workers for threaded evaluation')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Process records in batches to avoid lasj timeout (fix batch size)')
    
    args = parser.parse_args()

    # print configuration
    from pprint import pprint
    print(f"Configuration:\n")
    pprint(args.__dict__)
    
    # Determine output path
    overwrite = False
    if args.output_path is None:
        overwrite = True
        args.output_path = args.results_path
        
    # Initialize evaluator
    judge_utils = JudgeUtils(base_url=args.base_url, 
        openai_api_key=args.openai_api_key, 
        judge_model_name=args.judge_model_name
    )
        
    # Load results
    print(f"Loading results from: {args.results_path}")
    results = []
    try:
        with open(args.results_path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {args.results_path}")
        sys.exit(1)

    
    if results and 'pred_output' in results[0]:
        assert args.n <= len(results[0]['pred_output']), (
            f"args.n={args.n} > len(pred_output)={len(results[0]['pred_output'])}"
        )
        
    print(f"Processing {len(results)} samples...")
    results, result_dict = eval_all_records(results, judge_utils, args) # NOTE: Main!
    print("Evaluation Results:")
    pprint(result_dict)

    # for vstar, apart from all, we need to calculate two different subsets:
    # 1. direct_attributes
    # 2. relative_position
    if "vstar" in args.results_path:
        subsets = ["direct_attribute", "relative_position"]
        for subset in subsets:
            results_subset = [res for res in results if subset in res['index']]
            print("="* 20 + f"\tV*_{subset}\t" + "="* 20)
            results_subset, result_dict_subset = eval_all_records(results_subset, judge_utils, args)
            pprint(result_dict_subset)

    # TreeBench: per-category breakdown and Mean IoU
    elif 'treebench' in args.results_path:
        for record in results:
            iou_per_traj = []
            for traj_idx, traj_info in enumerate(record['traj_info']):
                pred_flatten_bboxes_mapped_to_orig = extract_boxes_from_bboxes_list(
                    bboxes_list=traj_info['bboxes_list'],
                    scale=traj_info['scale']
                )
            target_bboxes = record['target_instances']
            iou_per_traj.append(
                _calculate_average_iou(pred_flatten_bboxes_mapped_to_orig, target_bboxes)
            )
            record['iou'] = iou_per_traj 
        mean_iou = calculate_mean_iou(results, args.n)
        print(f"==> Mean IoU: {mean_iou * 100:.2f}%")


        # get the fine-grained acc for each category
        TREEBENCH_TAGS = [
            "Perception/Attributes", "Perception/Material", "Perception/Physical State",
            "Perception/Object Retrieval", "Perception/OCR",
            "Reasoning/Perspective Transform", "Reasoning/Ordering", "Reasoning/Contact and Occlusion",
            "Reasoning/Spatial Containment", "Reasoning/Comparison",
        ]
        total = 0
        for tag in TREEBENCH_TAGS:
            subset = [r for r in results if r.get('category') == tag]
            if not subset:
                print("[WARNING] No samples found for category: ", tag)
                continue
            # re-calculate the acc@k and pass@k for each category
            # since `eval_all_records` will not recompute if the correct list is already computed
            # the current one line will results in the least computational overhead
            results_subset, result_dict_subset = eval_all_records(subset, judge_utils, args)
            print("="* 20 + f"\t{tag}\t" + "="* 20)
            pprint(result_dict_subset) 

    # Save modified results
    if overwrite:
        print("[WARNING] Overwriting the original results!")
    try:
        with open(args.output_path, 'w') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print("Done.")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
