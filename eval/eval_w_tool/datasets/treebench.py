"""
TreeBench dataset handler.
Processes TreeBench TSV files and outputs unified format.
"""
import os
import pandas as pd
import base64
import json
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .base import DatasetBase


class TreeBenchDataset(DatasetBase):
    """Handler for TreeBench dataset (TSV format)."""
    
    def __init__(self, dataset_path: str, image_dir: str = None):
        """
        Initialize TreeBench dataset handler.
        
        Args:
            dataset_path: Path to the TreeBench TSV file or directory containing TreeBench.tsv
            image_dir: Directory to save decoded images (optional)
        """
        # If dataset_path is a directory, look for TreeBench.tsv inside
        if os.path.isdir(dataset_path):
            tsv_path = os.path.join(dataset_path, "TreeBench.tsv")
            if os.path.exists(tsv_path):
                dataset_path = tsv_path
            else:
                raise FileNotFoundError(f"TreeBench.tsv not found in directory: {dataset_path}")
        
        super().__init__(dataset_path)
        self.image_dir = image_dir
        self._data = None
        
        # Set up image directory
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
        else:
            # Default: create images directory next to TSV file
            self.image_dir = os.path.join(os.path.dirname(self.dataset_path), "images")
            os.makedirs(self.image_dir, exist_ok=True)
        
        print("[TreeBenchDataset] saving intermediate images to: ", self.image_dir)

    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load TreeBench dataset from TSV file."""
        if self._data is not None:
            return self._data
        
        # Read TSV file
        df = pd.read_csv(self.dataset_path, sep='\t')
        total_rows = len(df)
        
        # Pre-extract option columns (A through K)
        option_cols = [col for col in df.columns if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']]
        has_image_col = 'image' in df.columns
        
        # Extract basic fields using vectorized operations
        indices = df.get('index', pd.Series(range(total_rows))).tolist()
        questions = df['question'].astype(str).tolist()
        answers = df.get('answer', pd.Series([''] * total_rows)).astype(str).tolist()
        categories = df.get('category', pd.Series([''] * total_rows)).astype(str).tolist()
        l2_categories = df.get('l2-category', pd.Series([''] * total_rows)).astype(str).tolist()
        
        # Extract target_instances (bboxes) - may be string representation of list
        target_instances_list = []
        if 'target_instances' in df.columns:
            for idx in range(total_rows):
                val = df.iloc[idx, df.columns.get_loc('target_instances')]
                if pd.notna(val):
                    try:
                        # Try to parse as JSON if it's a string
                        if isinstance(val, str):
                            target_instances_list.append(json.loads(val))
                        else:
                            target_instances_list.append(val)
                    except (json.JSONDecodeError, TypeError):
                        target_instances_list.append(None)
                else:
                    target_instances_list.append(None)
        else:
            target_instances_list = [None] * total_rows
        
        # Extract options for each row
        all_options = []
        if option_cols:
            # Get column positions for faster access
            col_positions = {col: df.columns.get_loc(col) for col in option_cols if col in df.columns}
            # Use itertuples for faster iteration
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
        
        if has_image_col:
            # Prepare image decoding tasks
            image_tasks = []
            for idx in range(total_rows):
                image_val = df.iloc[idx, df.columns.get_loc('image')]
                if pd.notna(image_val):
                    img_idx = indices[idx] if idx < len(indices) else idx
                    image_tasks.append((idx, str(image_val), img_idx))
            
            if image_tasks:
                # Use ProcessPoolExecutor for parallel image decoding
                max_workers = max(8, os.cpu_count())
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all image decoding tasks (pass image_dir as parameter for serialization)
                    future_to_idx = {
                        executor.submit(self._decode_and_save_image, img_data, img_idx, self.image_dir): idx
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
        
        # Build final data list
        data = []
        with tqdm(total=total_rows, desc="Loading TreeBench dataset", unit="samples") as pbar:
            for idx in range(total_rows):
                sample = {
                    'index': indices[idx] if idx < len(indices) else idx,
                    'question': questions[idx],
                    'options': all_options[idx],
                    'answer': answers[idx],
                    'category': categories[idx],
                    'l2-category': l2_categories[idx],
                    'target_instances': target_instances_list[idx],
                    'image_path': image_paths[idx],
                }
                data.append(sample)
                pbar.update(1)
        
        self._data = data
        return data
    
    @staticmethod
    def _decode_and_save_image(base64_image: str, index: Any, image_dir: str) -> str:
        """Decode base64 image and save to disk."""
        try:
            # Decode base64
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            
            # Save image
            image_filename = f"image_{index}.jpg"
            image_path = os.path.join(image_dir, image_filename)
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
    # Testing
    dataset_path = "/map-vepfs/haozhe/yhwu/vlmpaper/data/benchmarks/TreeBench"
    image_dir = os.path.join(dataset_path, "images")
    
    print(f"Loading TreeBench dataset from: {dataset_path}")
    ds = TreeBenchDataset(dataset_path, image_dir=image_dir)
    data = ds.load_data()
    
    for i in range(len(data)):
        if len(data[i]['options']) > 4:
            print(f"\n✓ Loaded {len(data)} samples")
            print(f"\nSample 0:")
            print(f"  index: {data[i]['index']}")
            print(f"  question: {data[i]['question'][:100]}...")
            print(f"  options: {data[i]['options']}")
            print(f"  answer: {data[i]['answer']}")
            print(f"  category: {data[i]['category']}")
            print(f"  l2-category: {data[i]['l2-category']}")
            print(f"  target_instances: {data[i]['target_instances']}")
            print(f"  image_path: {data[i]['image_path']}")
            
            print(f"\nFormatted prompt for sample 0:")
            print(ds.format_prompt(data[i]))
