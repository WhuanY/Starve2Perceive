"""
HRBench dataset handler.
Processes HRBench TSV files and outputs unified format.
"""
import os
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from functools import partial

from .base import DatasetBase
from .utils import load_jsonl


class GeneralVQADataset(DatasetBase):
    """Handler for General VQA dataset.
    """
    
    def __init__(self, dataset_path: str, image_dir: str = None):
        """
        Initialize General VQA dataset handler.
        
        Args:
            dataset_path: Path to the General VQA dataset
            image_dir: Directory to save decoded images (optional)
        """
        super().__init__(dataset_path)
        
        if image_dir is None:
            self.image_dir = os.path.join(dataset_path, "images")
        else:
            self.image_dir = image_dir

        print(self.image_dir)
        self._data = None
        
        if self.image_dir:
            os.makedirs(self.image_dir, exist_ok=True)
            print(" [GeneralVQADataset] Using existing image directory: ", self.image_dir)
        else:
            raise ValueError(f"image_dir for {dataset_path} is required")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load General VQA dataset from jsonl file.
        expect a dataset_path containing:
        - `*.jsonl`: a jsonl file containing the question-answer-image triplets.
        -  `images/`: a directory containing the images. The image file name should be the same as the image_path in the jsonl file.
        """
        if self._data is not None:
            return self._data

        # Read jsonl file
        if os.path.isdir(self.dataset_path):
            jsonl_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.jsonl')]
            if not jsonl_files:
                raise ValueError(f"No jsonl file found in directory: {self.dataset_path}")
            self._data = []
            for jsonl_file in jsonl_files:
                data = load_jsonl(os.path.join(self.dataset_path, jsonl_file))
                self._data.extend(data)
        else:
            raise ValueError(f"By default, GeneralVQADataset expects a directory path. Got: {self.dataset_path}")

        
        # change the image path to absolute path and add index field
        for idx, item in enumerate(data):
            image_filename = os.path.basename(item.get('image_path', "") or item.get('image', ""))
            item['image_path'] = os.path.join(self.image_dir, image_filename)
            # Add index field if not present (use id or enumerate index)
            if 'index' not in item:
                item['index'] = item.pop('id')
            assert os.path.exists(item['image_path']), f"Image not found: {item['image_path']}"
        
        self._data = data
        return data
    
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format prompt with question and options."""
        question = sample['question']
        options = sample.get('options', [])
        
        # Format options as A, B, C, D...
        if options:
            option_str = "\n"
            abc_map = {i: chr(65 + i) for i in range(26)}  # 0->A, 1->B, ...
            
            for i, option in enumerate(options):
                option_str += f"{abc_map[i]}. {option}\n"
            
            prompt = f"Question: {question}\nOptions: {option_str}"
        else:
            prompt = f"Question: {question}"
        return prompt


if __name__ == "__main__":
    # testing
    print("hello")
    ds = GeneralVQADataset("/root/autodl-tmp/benchmarks/realworldqa")
    import random
    rand_idx = random.randint(0, len(ds)-1)
    for i, d in enumerate(ds):
        if i == rand_idx:
            print(i, d)
            print(ds.format_prompt(d))
            print("-"*100)
            break
    