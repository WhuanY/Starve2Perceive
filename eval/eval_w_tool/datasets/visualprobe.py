"""
VisualProbe dataset handler.
Processes VisualProbe JSON files and outputs unified format.
"""
import os
import json
from typing import List, Dict, Any
from tqdm import tqdm

from .base import DatasetBase


class VisualProbeDataset(DatasetBase):
    """Handler for VisualProbe dataset (JSON format)."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize VisualProbe dataset handler.
        
        Args:
            dataset_path: Path to the VisualProbe dataset directory containing
                         train.json or val.json and data/ directory
        """
        super().__init__(dataset_path)
        self._data = None
        
        # Determine which JSON file to use (train.json or val.json)
        train_json = os.path.join(dataset_path, "train.json")
        val_json = os.path.join(dataset_path, "val.json")
        
        if os.path.exists(train_json):
            self.json_file = train_json
        elif os.path.exists(val_json):
            self.json_file = val_json
        else:
            raise FileNotFoundError(f"Neither train.json nor val.json found in {dataset_path}")
        
        # Image directory
        self.image_dir = os.path.join(dataset_path, "data")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load VisualProbe dataset from JSON file."""
        if self._data is not None:
            return self._data
        
        # Load JSON file
        with open(self.json_file, 'r') as f:
            raw_data = json.load(f)
        
        data = []
        
        with tqdm(total=len(raw_data), desc="Loading VisualProbe dataset", unit="samples") as pbar:
            for item in raw_data:
                # Extract question from problem (remove <image>\n prefix if present)
                problem = item.get('problem', '')
                if problem.startswith('<image>\n'):
                    question = problem[len('<image>\n'):]
                else:
                    print(f"[WARNING] No <image> prefix found in problem: {problem}")
                    question = problem
                
                # Get image path(s)
                images = item.get('images', [])
                assert len(images) == 1, "VisualProbe dataset should have only one image per sample"
                if not images:
                    print(f"Warning: No images found for doc_id {item.get('doc_id', 'unknown')}")
                    kill
                
                # Get the first image (assuming single image per sample)
                image_rel_path = images[0]
                # Extract filename from relative path (e.g., "VisualProbe_train/data/visual_probe_train_7.jpg" -> "visual_probe_train_7.jpg")
                image_filename = os.path.basename(image_rel_path)
                image_path = os.path.join(self.image_dir, image_filename)
                
                # Verify image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Build sample
                sample = {
                    'index': item.get('doc_id', ''),
                    'question': question,
                    'options': [],  # VisualProbe doesn't have multiple choice options
                    'answer': item.get('solution', ''),
                    'category': item.get('data_source', 'visual_probe'),  # Use data_source as category
                    'cycle_category': image_path,  # Use image path as cycle_category
                    'image_path': image_path,
                }
                data.append(sample)
                pbar.update(1)
        
        self._data = data
        return data
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format prompt with question and options."""
        question = sample['question']
        options = sample.get('options', [])
        
        # Format options as A, B, C, D... if they exist
        if options:
            print("[WARNING] VisualProbe dataset has options. I remember not!")
            kill
            option_str = "\n"
            abc_map = {i: chr(65 + i) for i in range(26)}  # 0->A, 1->B, ...
            
            for i, option in enumerate(options):
                option_str += f"{abc_map[i]}. {option}\n"
            
            prompt = f"Question: {question}\nOptions: {option_str}"
        else:
            # No options, just the question
            prompt = f"Question: {question}"
        
        return prompt


if __name__ == "__main__":
    # testing
    print("hello")
    # ds = VisualProbeDataset("/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_train")
    # ds = VisualProbeDataset("/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_Easy")
    ds = VisualProbeDataset("/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_Medium")
    ds = VisualProbeDataset("/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_Hard")
    data = ds.load_data()
    # print("data: ", data)
    # print("len(data): ", len(data))
    # print("data[0]: ", data[0])
    # kill
    # print("data[0]['question']: ", data[0]['question']) # What is the number displayed above the entrance where the woman is standing?
    # print("data[0]['options']: ", data[0]['options']) # ['27B', '37B', '27D', '27E']
    # print("data[0]['answer']: ", data[0]['answer']) # A
    # print("data[0]['category']: ", data[0]['category']) # single
    # print("data[0]['cycle_category']: ", data[0]['cycle_category']) # /tmp/hrbench_images__sako2vq/image_0.jpg
    # print("data[0]['image_path']: ", data[0]['image_path']) # 
    for d in data:
        print(d)
        break

    print("yes")
