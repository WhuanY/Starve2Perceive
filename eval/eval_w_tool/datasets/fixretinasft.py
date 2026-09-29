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


class FixRetinaSFTDataset(DatasetBase):
    """Handler for FixRetina SFT dataset."""
    
    def __init__(self, dataset_path: str, image_dir: str = None):
        """
        Initialize FixRetina SFT dataset handler.
        
        Args:
            dataset_path: Path to the FixRetina SFT dataset
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
            print("[FixRetinaSFTDataset] Using existing image directory: ", image_dir)
        else:
            raise ValueError("image_dir for fixretina_sft is required")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load FixRetina SFT dataset from jsonl file."""
        if self._data is not None:
            return self._data

        # Read jsonl file
        data = load_jsonl(os.path.join(self.dataset_path, "fix_retina_sft_qa.jsonl"))
        
        # change the image path to absolute path and add index field
        for idx, item in enumerate(data):
            item['image_path'] = os.path.join(self.image_dir, item['image_path'])
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
    ds = FixRetinaSFTDataset("/home/ywuit/vlmpaper/data/FixRetina_SFT/sft")
    for d in ds:
        print(d)