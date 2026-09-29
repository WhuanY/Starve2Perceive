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
from .utils import load_json


class MMERealWorldLiteDataset(DatasetBase):
    """Handler for MMERealWorldLite dataset."""
    
    def __init__(self, dataset_path: str, image_dir: str = None):
        """
        Initialize MMERealWorldLite dataset handler.
        
        Args:
            dataset_path: Path to the FixRetina SFT dataset
            image_dir: Directory to save decoded images (optional)
        """
        super().__init__(dataset_path)
        
        if image_dir is None:
            self.image_dir = os.path.join(dataset_path, "imgs")
        else:
            self.image_dir = image_dir

        print(self.image_dir)
        assert os.path.exists(self.image_dir), f"Image directory not found: {self.image_dir}"
        self._data = None
    
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load FixRetina SFT dataset from jsonl file."""
        if self._data is not None:
            return self._data

        # Read json file
        data = load_json(os.path.join(self.dataset_path, "MME-RealWorld-Lite.json"))
        
        # change the image path to absolute path and add index field
        for idx, item in enumerate(data):
            if 'Question Type' in item:
                assert item.pop('Question Type') == 'Multiple Choice', f"{idx}" # MME-RealWorld-Lite only supports multiple choice questions
            elif 'Question type' in item:
                assert item.pop('Question type') == 'Multiple Choice', f"{idx}" # MME-RealWorld-Lite only supports multiple choice questions
            else:
                print(idx, item)
                kill
            item['image_path'] = os.path.join(self.image_dir, item.pop('Image'))
            item['question'] = item.pop('Text')
            abc_map = {i: chr(65 + i) for i in range(26)}
            item['options'] = [option.replace(f"({abc_map[i]})", "").strip() for i, option in enumerate(item.pop('Answer choices'))]
            # Add index field if not present (use id or enumerate index)
            if 'index' not in item:
                item['index'] = item.pop('Question_id')
            item['answer'] = item.pop('Ground truth')
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
    ds = MMERealWorldLiteDataset("/root/autodl-tmp/benchmarks/mmerealworldlite")
    for i, d in enumerate(ds):
        if i == 1034:
            print(d)
            print(ds.format_prompt(d))
            break