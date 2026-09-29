"""
HRBench dataset handler.
Processes HRBench TSV files and outputs unified format.
"""
import os
import pandas as pd
import base64
import tempfile
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial

from .base import DatasetBase


class HRBenchDataset(DatasetBase):
    """Handler for HRBench dataset (TSV format)."""
    
    def __init__(self, dataset_path: str, image_dir: str = None):
        """
        Initialize HRBench dataset handler.
        
        Args:
            dataset_path: Path to the HRBench TSV file
            image_dir: Directory to save decoded images (optional)
        """
        super().__init__(dataset_path)
        self.image_dir = image_dir
        self._data = None
        
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
        else:
            if "4k" in self.dataset_path:
                self.image_dir = os.path.join(os.path.dirname(self.dataset_path), "images_4k")
            elif "8k" in self.dataset_path:
                self.image_dir = os.path.join(os.path.dirname(self.dataset_path), "images_8k")
            else:
                raise ValueError("image_dir for hrbench is required")
        
        os.makedirs(self.image_dir, exist_ok=True)
        print("[HRBenchDataset] saving intermediate images to: ", self.image_dir)

    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load HRBench dataset from TSV file."""
        if self._data is not None:
            return self._data
        
        # Read TSV file
        df = pd.read_csv(self.dataset_path, sep='\t')
        total_rows = len(df)
        
        # Pre-extract option columns for faster access
        option_cols = [col for col in df.columns if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']]
        has_image_col = 'image' in df.columns
        
        # Extract basic fields using vectorized operations (much faster than iterrows)
        indices = df.get('index', pd.Series(range(total_rows))).tolist()
        questions = df['question'].astype(str).tolist()
        answers = df.get('answer', pd.Series([''] * total_rows)).astype(str).tolist()
        categories = df.get('category', pd.Series([''] * total_rows)).astype(str).tolist()
        cycle_categories = df.get('cycle_category', pd.Series([None] * total_rows)).tolist()
        
        # Extract options for each row (using itertuples is faster than iterrows)
        all_options = []
        if option_cols:
            # Get column positions for faster access
            col_positions = {col: df.columns.get_loc(col) for col in option_cols if col in df.columns}
            # Use itertuples for faster iteration (much faster than iterrows)
            for row_tuple in df.itertuples(index=False):
                options = []
                for col, pos in col_positions.items():
                    val = row_tuple[pos]
                    if pd.notna(val):
                        options.append(str(val))
                all_options.append(options)
        else:
            all_options = [[]] * total_rows
        
        # Parallel image decoding
        image_paths = [None] * total_rows
        raw_images = [None] * total_rows
        
        if has_image_col:
            # Prepare image decoding tasks
            image_tasks = []
            for idx in range(total_rows):
                image_val = df.iloc[idx, df.columns.get_loc('image')]
                if pd.notna(image_val):
                    img_idx = indices[idx] if idx < len(indices) else idx
                    image_tasks.append((idx, str(image_val), img_idx))
                    raw_images[idx] = str(image_val)
            
            if image_tasks:
                # Use ThreadPoolExecutor for I/O-bound image decoding
                max_workers = max(8, os.cpu_count())
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all image decoding tasks
                    future_to_idx = {
                        executor.submit(self._decode_and_save_image, img_data, img_idx): idx
                        for idx, img_data, img_idx in image_tasks
                    }
                    
                    # Collect results with progress bar
                    with tqdm(total=len(image_tasks), desc="Decoding images", unit="images", leave=False) as pbar:
                        for future in as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                image_paths[idx] = future.result()
                            except Exception as e:
                                print(f"Error decoding image for index {idx}: {e}")
                                image_paths[idx] = None
                            pbar.update(1)
        
        # Build final data list (vectorized construction)
        data = []
        with tqdm(total=total_rows, desc="Loading HRBench dataset", unit="samples") as pbar:
            for idx in range(total_rows):
                sample = {
                    'index': indices[idx] if idx < len(indices) else idx,
                    'question': questions[idx],
                    'options': all_options[idx],
                    'answer': answers[idx],
                    'category': categories[idx],
                    'cycle_category': cycle_categories[idx],
                    'image_path': image_paths[idx],
                }
                data.append(sample)
                pbar.update(1)
        
        self._data = data
        return data
    
    def _decode_and_save_image(self, base64_image: str, index: Any) -> str:
        """Decode base64 image and save to disk."""
        try:
            # Decode base64
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            
            # Save image
            image_filename = f"image_{index}.jpg"
            image_path = os.path.join(self.image_dir, image_filename)
            image.save(image_path, format='JPEG')
            
            return image_path
        except Exception as e:
            print(f"Error decoding image for index {index}: {e}")
            return None
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format prompt with question and options."""
        question = sample['question']
        options = sample.get('options', [])
        
        # Format options as A, B, C, D...
        option_str = "\n"
        abc_map = {i: chr(65 + i) for i in range(26)}  # 0->A, 1->B, ...
        
        for i, option in enumerate(options):
            option_str += f"{abc_map[i]}. {option}\n"
        
        prompt = f"Question: {question}\nOptions: {option_str}"
        return prompt


if __name__ == "__main__":
    # testing
    print("hello")
    ds = HRBenchDataset("/home/ywuit/vlmpaper/data/hrbench/hr_bench_4k.tsv", "/home/ywuit/vlmpaper/data/hrbench/images")
    data = ds.load_data()
    # print("data: ", data)
    print("len(data): ", len(data))
    print("data[0]['question']: ", data[0]['question']) # What is the number displayed above the entrance where the woman is standing?
    print("data[0]['options']: ", data[0]['options']) # ['27B', '37B', '27D', '27E']
    print("data[0]['answer']: ", data[0]['answer']) # A
    print("data[0]['category']: ", data[0]['category']) # single
    print("data[0]['cycle_category']: ", data[0]['cycle_category']) # /tmp/hrbench_images__sako2vq/image_0.jpg
    print("data[0]['image_path']: ", data[0]['image_path']) # 
