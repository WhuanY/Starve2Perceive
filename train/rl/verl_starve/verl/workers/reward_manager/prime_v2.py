# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from collections import defaultdict
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, task_extra_info),  # Ensure synchronous
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        import traceback
        traceback.print_exc()
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print("shut down failed: " + str(kill_err))
            raise

    # Process results
    
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            scores.append({
                "score": 0.0,
                "acc_reward": 0.0,
                "format_reward": 0.0,
                "num_focused_regions": 0,
            }) # Note that this line is hardcoded for now.
        elif isinstance(result[0], (int, float, bool)):
            # Convert simple numeric result to dict format
            scores.append({
                "score": float(result[0]),
                "acc_reward": 0.0,
                "format_reward": 0.0,
                "num_focused_regions": 0,
            })
        elif isinstance(result, list) and len(result) > 0:
            # If result[0] is already a dict, use it; otherwise wrap it
            if isinstance(result[0], dict):
                scores.append(result[0])
            else:
                scores.append({
                    "score": float(result[0]) if isinstance(result[0], (int, float, bool)) else 0.0,
                    "acc_reward": 0.0,
                    "format_reward": 0.0,
                    "num_focused_regions": 0,
                })
        else:
            # Fallback: try to extract score from nested structure
            try:
                score_val = float(result[0][0]) if isinstance(result[0], (list, tuple)) else float(result[0])
                scores.append({
                    "score": score_val,
                    "acc_reward": 0.0,
                    "format_reward": 0.0,
                    "num_focused_regions": 0,
                })
            except (IndexError, TypeError, ValueError):
                scores.append({
                    "score": 0.0,
                    "acc_reward": 0.0,
                    "format_reward": 0.0,
                    "num_focused_regions": 0,
                })

    # print(f"[PrimeRewardManager] {scores=}", flush=True)
    return scores # List[dict]


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        """
        verify the batch asynchronously
        """
        # batched scoring
        # data.non_tensor_batch.keys()=dict_keys(['ability', 'data_source', 'reward_model', 'extra_info', 'index', 'uid', 'multi_modal_inputs'])
        # index: same index refer to the same question
        # uid: uid is the unique identifier for each trajectory.
        
        sequences_str = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # decode
            response_str = self.tokenizer.decode(valid_response_ids)
            sequences_str.append(response_str)
        
        # response_ids = data.batch["responses"] # (bs, response_length)
        # valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        # sequences_str = self.tokenizer.batch_decode(re sponse_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_compute_score_async(
                    self.compute_score,
                    sequences_str, # completion
                    ground_truth, # reference
                    data_sources, # task
                    extra_info=extra_info,
                    num_processes=64,
                )
            )
        except asyncio.TimeoutError:
            print("Global timeout in reward computing! Setting all as 0.")
            scores = [{
                "score": 0.0,
                "acc_reward": 0.0,
                "format_reward": 0.0,
                "num_focused_regions": 0,
            } for _ in range(len(sequences_str))]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [{
                "score": 0.0,
                "acc_reward": 0.0,
                "format_reward": 0.0,
                "num_focused_regions": 0,
            } for _ in range(len(sequences_str))]
        # data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]
        uids = data.non_tensor_batch["uid"]
        reward_extra_info = defaultdict(list)
        
        scores = self.verify(data) # parallel scoring for all trajectories

        uid2nfr = defaultdict(list)
        uid2acc = defaultdict(list)

        # 第一遍遍历：得到 idx2numFocusedRegion 和 idx2acc
        for i in range(len(data)):
            if isinstance(scores[i], dict):
                uid2nfr[uids[i]].append(scores[i]["num_focused_regions"])
                uid2acc[uids[i]].append(scores[i]["acc_reward"])

        print(f"{uid2acc=}")
        print(f"{uid2nfr=}")
        uid2maxFocusedRegionWhenCorrect = defaultdict(int)
        uid2minFocusedRegionWhenCorrect = defaultdict(int)
        for uid in uid2nfr.keys():
            maxnfr_correct = 0
            minfr_correct = 100 # a larger number
            for idx, nfr in enumerate(uid2nfr[uid]):
                if uid2acc[uid][idx] > 0:
                    maxnfr_correct = max(maxnfr_correct, nfr)
                    minfr_correct = min(minfr_correct, nfr)
            uid2maxFocusedRegionWhenCorrect[uid] = maxnfr_correct
            uid2minFocusedRegionWhenCorrect[uid] = minfr_correct
        print(f"{uid2maxFocusedRegionWhenCorrect=}")
        print(f"{uid2minFocusedRegionWhenCorrect=}")

        # 第二遍遍历：根据 uid2acc 和uid2numFocusedRegion的情况动态分配tool_reward
        for i in range(len(data)):
            data_source = data_sources[i]
            uid = uids[i]
            acc = scores[i]['acc_reward']
            fmt = scores[i]['format_reward']
            # 2 / 8 == 0.25， set higher for numerial stability
            if isinstance(scores[i], dict):
                for key, value in scores[i].items():
                    reward_extra_info[key].append(value)
                reward = scores[i]['score'] 
            else:
                reward = scores[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = reward
        
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
            
            response_ids_ = data[i].batch["responses"]
            valid_response_length_ = data[i].batch["attention_mask"][prompt_length:].sum()
            valid_response_ids_ = response_ids_[:valid_response_length_]
            response_str_ = self.tokenizer.decode(valid_response_ids_)
            with open("/map-vepfs/haozhe/yhwu/vlmpaper/tmp/check_reward/traj_reward.jsonl", "a") as f:
                f.write(json.dumps({
                    "uid": uid,
                    "question": data.non_tensor_batch.get("extra_info")[i].get("question"),
                    "options": data.non_tensor_batch.get('extra_info')[i].get("options"),
                    "response": sequences_str[i],
                    "score": scores[i]['score'],
                    "acc_reward": scores[i]['acc_reward'],
                    "format_reward": scores[i]['format_reward'],
                    "num_focused_regions": scores[i]['num_focused_regions'],
                    "cond_tool_reward": -1.0, 
                    "response_str": response_str_
                }) + "\n") # debug

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
