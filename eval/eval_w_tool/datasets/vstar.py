"""
VStar dataset handler.
Processes VStar JSON files and outputs unified format.
Expects directory structure:
    dataset_path/
        direct_attributes/
            image_0.jpg
            image_0.json
            ...
        relative_position/
            image_0.jpg
            image_0.json
            ...
"""
import os
import json
from typing import List, Dict, Any
from glob import glob
from tqdm import tqdm

from .base import DatasetBase


class VStarDataset(DatasetBase):
    """Handler for VStar dataset (JSON format)."""
    
    # VStar has two test types
    TEST_TYPES = ['direct_attributes', 'relative_position']
    
    def __init__(self, dataset_path: str):
        """
        Initialize VStar dataset handler.
        
        Args:
            dataset_path: Path to VStar dataset directory containing 
                          'direct_attributes' and 'relative_position' subdirectories
        """
        super().__init__(dataset_path)
        self._data = None
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load VStar dataset from directory."""
        if self._data is not None:
            return self._data
        
        data = []
        
        for test_type in self.TEST_TYPES:
            test_path = os.path.join(self.dataset_path, test_type)
            
            if not os.path.exists(test_path):
                print(f"Warning: Test type directory not found: {test_path}")
                continue
            
            # Find all files in the directory
            all_files = os.listdir(test_path)
            
            # Filter to get only image files (exclude .json files)
            image_files = [f for f in all_files if not f.endswith('.json')]
            
            print(f"Loading VStar [{test_type}]: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"Loading {test_type}"):
                img_path = os.path.join(test_path, img_file)
                
                # Find corresponding JSON annotation file
                # Remove extension and add .json
                base_name = os.path.splitext(img_file)[0]
                anno_path = os.path.join(test_path, base_name + '.json')
                
                if not os.path.exists(anno_path):
                    print(f"Warning: Annotation not found for {img_file}")
                    continue
                
                with open(anno_path, 'r') as f:
                    anno = json.load(f)
                
                sample = {
                    'index': f"{test_type}/{img_file}",  # Include test_type in index for uniqueness
                    'image': img_file,  # Original image filename
                    'question': anno['question'],
                    'options': anno['options'],
                    'answer': anno['options'][0] if anno['options'] else '',  # First option is correct
                    'image_path': img_path,
                    'test_type': test_type,  # Track which test type this belongs to
                    'bbox': anno.get('bbox', []),
                    'target_object': anno.get('target_object', []),
                }
                data.append(sample)
        
        self._data = data
        print(f"Total loaded: {len(data)} samples")
        return data
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format prompt with question and options."""
        question = sample['question']
        options = sample.get('options', [])
        
        # Format options as A, B, C, D...
        option_str = "\n"
        abc_map = {i: chr(65 + i) for i in range(26)}
        
        for i, option in enumerate(options):
            option_str += f"{abc_map[i]}. {option}\n"
        
        prompt = f"Question: {question}\nOptions: {option_str}"
        return prompt


if __name__ == "__main__":
    # testing
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/root/autodl-tmp/benchmarks/vstar_bench"
    
    print(f"Testing VStarDataset with path: {dataset_path}")
    ds = VStarDataset(dataset_path)
    data = ds.load_data()
    
    print(f"\nTotal samples: {len(data)}")
    
    if data:
        print(f"\nFirst sample:")
        print(f"  index: {data[0]['index']}")
        print(f"  test_type: {data[0]['test_type']}")
        print(f"  question: {data[0]['question']}")
        print(f"  options: {data[0]['options']}")
        print(f"  answer: {data[0]['answer']}")
        print(f"  image_path: {data[0]['image_path']}")
        # format
        print(f"  formatted prompt: {ds.format_prompt(data[0])}")
