"""
Main inference engine with tool calling support.
Handles model inference, tool execution, and result saving.
"""
import os
import json
import math
from io import BytesIO
import base64 as b64
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import asyncio
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from PIL import Image
from openai import AsyncOpenAI

try:
    # Try relative imports first (when used as a package)
    from ..config import Config
    from ..datasets.base import DatasetBase
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config import Config
    from datasets.base import DatasetBase

from .image_utils import (
    encode_image_to_base64,
    encode_pil_image_to_base64,
    smart_resize,
    IMAGE_FACTOR
)
from .prompts import (
    INSTRUCTION_PROMPT_SYSTEM,
    USER_PROMPT_TEMPLATE,
    AFTER_PROMPT,
    TOOL_CALL_START_TOKEN,
    TOOL_CALL_END_TOKEN,
    ANSWER_START_TOKEN,
    ANSWER_END_TOKEN
)

# Use AFTER_PROMPT as USER_PROMPT_COMMON (they are the same)
USER_PROMPT_COMMON = AFTER_PROMPT


class InferenceEngine:
    """Main inference engine for VQA with tool calling."""
    
    def __init__(self, config: Config):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_url,
        )
        self.eval_model_name = config.eval_model_name
        if self.eval_model_name is None:
            print(f"[InferenceEngine] eval_model_name is None, use model_name: {config.model_name}")
            self.eval_model_name = config.model_name
        self.sample_n = config.sample_n
    
    async def infer_sample(self, sample: Dict[str, Any], dataset: DatasetBase, sample_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Infer a single sample with tool calling support.
        
        Args:
            sample: Data sample from dataset
            dataset: Dataset handler instance
            sample_n: Number of samples to infer
            
        Returns:
            Dictionary containing inference results
        """

        # Use provided sample_n or default from config
        if sample_n is None:
            sample_n = self.sample_n if self.sample_n is not None else 1
        
        result_dict = {
            "index": sample.get('index'),
            "question": sample.get('question'),
            "options": sample.get('options', None),
            'answer': sample.get('answer'),
            "pred_ans": [None] * sample_n,
            "pred_output": [None] * sample_n,
            "status": [None] * sample_n,
        }

        async def single_inference_with_index(idx_n: int):
            output_text, print_messages, status = await self._async_rollout_a_request(dataset, sample)
            return (output_text, print_messages, status)

        tasks = [single_inference_with_index(idx_n) for idx_n in range(sample_n)]

        sample_results = await asyncio.gather(*tasks)

        for idx_n, (output_text, print_messages, status) in enumerate(sample_results):
            result_dict["pred_ans"][idx_n] = output_text
            result_dict["pred_output"][idx_n] = print_messages
            result_dict["status"][idx_n] = status
        
        return result_dict
        
    
    async def infer_dataset(
        self, 
        dataset: DatasetBase,
        data_slice: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        num_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a list of samples asynchronously.
        
        Args:
            dataset: Dataset handler
            data_slice: List of data samples to process
            output_path: Path to save results (JSONL format)
            num_workers: Number of concurrent workers (uses config if None)
            
        Returns:
            List of inference results
        """
        num_workers = num_workers or self.config.num_workers
        
        num_samples = len(data_slice)
        
        # ======== Async Inference =========
        print(f"Processing {num_samples} samples")
        print(f"Using {num_workers} concurrent workers")
        
        semaphore = asyncio.Semaphore(num_workers)
        
        async def process_with_index(idx: int, sample: Dict[str, Any]) -> tuple[int, Optional[Dict[str, Any]]]:
            """Process a sample and return its index along with the result."""
            async with semaphore:
                try:
                    result = await self.infer_sample(sample, dataset, sample_n=self.sample_n)
                    return (idx, result)
                except Exception as e:
                    print(f"Error processing sample {sample.get('index', 'unknown')}: {e}")
                    return (idx, None)
        
        batch_size = 4  # async with fix batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size

        print(f"Batch size: {batch_size}, Total batches: {num_batches}")

        # Create directory if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            print(f"Saving results to {output_path}...")

        all_results = []
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Processing batches"): # control async tasks number
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data_slice[start_idx:end_idx]

            tasks = [process_with_index(start_idx + idx_in_batch, batch_data[idx_in_batch]) for idx_in_batch in range(len(batch_data))]
            batch_results_with_idx = await asyncio.gather(*tasks)
            
            # Extract results from (idx, result) tuples
            batch_results = []
            for idx, result in batch_results_with_idx:
                if result is not None:
                    batch_results.append(result)
                    all_results.append(result)
            
            # Save batch results immediately to file (overwrite mode for specific indices)
            if output_path and batch_results:
                self._save_batch_results(output_path, batch_results)
        
        print(f"✓ Inference complete. Processed {len(all_results)} samples.")
        return all_results

    def _save_batch_results(self, output_path: str, batch_results: List[Dict[str, Any]]):
        """
        Save a batch of results to the output file.
        Updates existing records if they exist (based on 'index'), otherwise appends.
        
        Args:
            output_path: Path to the JSONL file
            batch_results: List of result dictionaries to save
        """
        # Read existing data
        existing_data = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                existing_data[item['index']] = item
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"Error reading existing results for update: {e}")
        
        # Update with new results
        for result in batch_results:
            existing_data[result['index']] = result
            
        # Write all back to file (sorted by index)
        try:
            sorted_indices = sorted(existing_data.keys())
            with open(output_path, 'w') as f:
                for idx in sorted_indices:
                    f.write(json.dumps(existing_data[idx], ensure_ascii=False) + '\n')
        except Exception as e:
             print(f"Error saving batch results: {e}")

    
    async def _async_rollout_a_request(self, dataset: DatasetBase, sample: Dict[str, Any]) -> tuple[str, list[dict], str]:
        """
        Execute a single inference request with tool calling support.
        
        This method performs one complete inference rollout, which may include:
        - Initial image encoding
        - Multi-turn conversation with tool calls
        - Image cropping and re-analysis
        - Answer extraction
        
        Args:
            sample: Data sample containing question and image info
            dataset: Dataset handler for prompt formatting
            
        Returns:
            tuple: (output_text, print_messages, status)
                - output_text: Extracted answer string
                - print_messages: Conversation history for logging
                - status: 'success' or 'error'
        """
        # Get image path(s)
        image_path = sample.get('image_path', "")
        
        # Handle single image path
        if isinstance(image_path, str) and image_path != "":
            if os.path.exists(image_path): # 首先尝试读取本地图片
                # print("[DEBUG] image_path exists")
                pil_img = Image.open(image_path)
                base64_image = encode_image_to_base64(image_path)
            elif sample.get('raw_image'):  # 如果本地图片不存在，尝试读取base64编码的图片
                print("[DEBUG] sample.get('raw_image') exists")
                base64_image = sample['raw_image']
                image_data = b64.b64decode(base64_image)
                pil_img = Image.open(BytesIO(image_data))
            else:
                print("else")
                if sample.get('raw_image'):
                    base64_image = sample['raw_image']
                    image_data = b64.b64decode(base64_image)
                    pil_img = Image.open(BytesIO(image_data))
                else:
                    raise ValueError(f"Image not found: {image_path}")
        else:
            assert 0 == 1, "BUG!!!, image_path"

        
        # Format prompt
        dataset_prompt = dataset.format_prompt(sample)
        # Extract question from dataset_prompt or use sample question
        user_prompt = USER_PROMPT_TEMPLATE.format(dataset_prompt=dataset_prompt, AFTER_PROMPT=AFTER_PROMPT)
        # Build initial messages with system prompt for tool calling
        messages = [
            {
                "role": "system",
                "content": INSTRUCTION_PROMPT_SYSTEM
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        # For saving (without actual image data)
        print_messages = [
            {
                "role": "system",
                "content": INSTRUCTION_PROMPT_SYSTEM
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        chat_message = messages
        response_message = ""
        status = 'success'
        try_count = 0
        
        try:
            # Main inference loop
            #print(f"[DEBUG] Main inference loop, try_count = {try_count}")
            while ANSWER_END_TOKEN not in response_message:
                if ANSWER_END_TOKEN in response_message and ANSWER_START_TOKEN in response_message:
                    #print(f"[DEBUG] ANSWER_END_TOKEN in response_message and ANSWER_START_TOKEN in response_message, break")
                    break
                
                if try_count > 3:
                    #print(f"[DEBUG] try_count > 3, break")
                    break
                
                # API call
                params = {
                    "model": self.eval_model_name, # openai api server requires this field.
                    "messages": chat_message,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stop": ["<|im_end|>\n".strip(), TOOL_CALL_END_TOKEN],
                }
                #print(f"[DEBUG] {try_count=}")
                #print(f"[DEBUG] print_messages at try_count {try_count} ", print_messages)
                # await asyncio.sleep(0.1)  # Small delay to avoid overwhelming the API
                response = await self.client.chat.completions.create(**params)
                response_message = response.choices[0].message.content
                #print(f"[DEBUG] response_message at try_count {try_count}: ", response_message)
                
                # Handle incomplete tool calls
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                if finish_reason == 'stop' and TOOL_CALL_START_TOKEN in response_message and TOOL_CALL_END_TOKEN not in response_message:
                    response_message += TOOL_CALL_END_TOKEN
                
                # Handle tool calling
                if TOOL_CALL_START_TOKEN in response_message:
                    # Extract tool call
                    tool_call_json = response_message.split(TOOL_CALL_START_TOKEN)[1].split(TOOL_CALL_END_TOKEN)[0].strip()
                    try:
                        action_list = json.loads(tool_call_json)
                        #print(f"[DEBUG] tool_call_json parsed via json")
                    except:
                        action_list = eval(tool_call_json)  # Fallback for malformed JSON
                        #print(f"[DEBUG] tool_call_json Fallback to eval")

                    
                    # Execute tool (image zoom)
                    bbox = action_list['arguments']['bbox_2d']
                    left, top, right, bottom = bbox
                    
                    # Crop image
                    cropped_image = pil_img.crop((left, top, right, bottom))
                    
                    # Smart resize
                    new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                    
                    # Encode cropped image
                    cropped_base64 = encode_pil_image_to_base64(cropped_image)
                    
                    # Build tool response
                    content_f = [
                        {"type": "text", "text": "<tool_response>"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_base64}"}},
                        {"type": "text", "text": user_prompt},
                        {"type": "text", "text": "</tool_response>"}
                    ]
                    
                    # Add to conversation
                    chat_message.extend([
                        {"role": "assistant", "content": response_message},
                        {"role": "user", "content": content_f}
                    ])
                    
                    print_messages.extend([
                        {"role": "assistant", "content": response_message},
                        {"role": "user", "content": [
                            {"type": "text", "text": "<tool_response>"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_prompt},
                            {"type": "text", "text": "</tool_response>"}
                        ]}
                    ])
                else:
                    # No tool call - add response to conversation history
                    chat_message.append({"role": "assistant", "content": response_message})
                    print_messages.append({"role": "assistant", "content": response_message})
                
                try_count += 1
                
        except Exception as e:
            print(f"Error processing sample {sample.get('index', 'unknown')}: {e}")
            status = 'error'
        
        # Extract answer
        if ANSWER_END_TOKEN in response_message and ANSWER_START_TOKEN in response_message:
            output_text = response_message.split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0].strip()
        else:
            output_text = response_message
        
        return output_text, print_messages, status
        

if __name__ == "__main__":
    async def test():
        from datasets.hrbench import HRBenchDataset
        dataset = HRBenchDataset("/root/autodl-tmp/benchmarks/hrbench/hr_bench_4k.tsv", image_dir="/root/autodl-tmp/benchmarks/hrbench/images")
        # print("Loading dataset samples...")
        # data = dataset.load_data()
        
        engine = InferenceEngine(
            config=Config(
                api_key="none", 
                api_url="http://localhost:8000/v1",
                model_name="Qwen2.5-VL-3B-Instruct",
                dataset_name="HRBench4K",
                dataset_path="/home/ywuit/vlmpaper/data/hrbench/hr_bench_4k.tsv",
                save_path="/home/ywuit/vlmpaper/data/hrbench/hr_bench_4k_results",
                eval_model_name="Qwen2.5-VL-3B-Instruct",
                num_workers=1,
                max_tokens=10240,
                temperature=0.0,
            )
        )
        
        # result = await engine.infer_sample(data[0], dataset) # 测试推理单条记录
        sample = {
            'question': 'What is the number displayed above the entrance where the woman is standing?',
            'options': ['27B', '37B', '27D', '27E'],
            'answer': 'A',
            'category': 'single',
            'cycle_category': '/root/autodl-tmp/benchmarks/hrbench/images/image_0.jpg',
            'image_path': '/root/autodl-tmp/benchmarks/hrbench/images/image_0.jpg',
        }
        result = await engine.infer_sample(sample, dataset)
        print("result: ", result)
    
    asyncio.run(test())
