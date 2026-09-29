"""
TreeVGR RL dataset handler.
Processes TreeVGR Parquet files (images, problem, answer, target_instances) and outputs unified format.
All questions are open-ended (not multiple-choice).
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from .base import DatasetBase


class TreeVGRRLDataset(DatasetBase):
    """Handler for TreeVGR RL dataset (Parquet format)."""

    def __init__(self, dataset_path: str):
        """
        Initialize TreeVGR RL dataset handler.

        Args:
            dataset_path: Path to the Parquet file (e.g. vstar30k_visdrone6k_x1y1x2y2.parquet)
        """
        if not dataset_path.endswith(".parquet") and os.path.isdir(dataset_path):
            parquet_path = os.path.join(dataset_path, "vstar30k_visdrone6k_x1y1x2y2.parquet")
            if os.path.exists(parquet_path):
                dataset_path = parquet_path
            else:
                raise FileNotFoundError(f"Parquet not found in directory: {dataset_path}")

        super().__init__(dataset_path)
        self._data = None
        self._base_dir = os.path.dirname(os.path.abspath(self.dataset_path))

    def load_data(self, start_index: int = 0, end_index: int = -1) -> List[Dict[str, Any]]:
        """Load TreeVGR RL dataset from Parquet file."""
        if self._data is not None and start_index == 0 and end_index == -1:
            return self._data

        df = pd.read_parquet(self.dataset_path)
        total_rows = len(df)

        if end_index < 0:
            end_index = total_rows
        df_slice = df.iloc[start_index:end_index]
        n_slice = len(df_slice)

        data = []
        for idx in tqdm(range(n_slice), desc="Loading TreeVGR RL dataset", unit="samples"):
            row_idx = start_index + idx
            row = df_slice.iloc[idx]

            # images: array like ['images/0.jpg'] -> resolve to absolute path(s)
            images_val = row.get("images", None)
            if images_val is not None:
                arr = np.asarray(images_val)
                paths = [
                    os.path.join(self._base_dir, p) if not os.path.isabs(str(p)) else str(p)
                    for p in (arr.tolist() if arr.size else [])
                ]
                image_path = paths[0] if len(paths) == 1 else (paths if paths else None)
            else:
                image_path = None

            # problem: '<image>\nQuestion text...' -> question (text only), problem (full prompt)
            problem = row.get("problem", "")
            if pd.isna(problem):
                problem = ""
            problem = str(problem).strip()
            if problem.startswith("<image>\n"):
                question = problem[len("<image>\n"):].strip()
            else:
                question = problem

            # answer
            answer_val = row.get("answer", "")
            answer = "" if pd.isna(answer_val) else str(answer_val)

            # target_instances: array of {bbox, label, name} -> list of dicts
            ti_val = row.get("target_instances", None)
            if ti_val is not None and (not isinstance(ti_val, np.ndarray) or len(ti_val) > 0):
                arr = np.asarray(ti_val, dtype=object)
                target_instances = []
                for item in (arr.tolist() if arr.size else []):
                    if hasattr(item, "item"):
                        item = item.item()
                    d = item if isinstance(item, dict) else {}
                    target_instances.append({
                        "bbox": list(d.get("bbox", [])),
                        "label": d.get("label"),
                        "name": d.get("name"),
                    })
            else:
                target_instances = None

            sample = {
                "index": row_idx,
                "question": question,
                "problem": problem,
                "answer": answer,
                "target_instances": target_instances,
                "image_path": image_path,
            }
            data.append(sample)

        if start_index == 0 and end_index == -1:
            self._data = data
        return data

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format prompt from sample. Uses full problem (with <image> placeholder); no options (open-ended)."""
        return sample.get("problem", sample.get("question", ""))


if __name__ == "__main__":
    dataset_path = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/TreeVGR-RL-37K/vstar30k_visdrone6k_x1y1x2y2.parquet"

    ds = TreeVGRRLDataset(dataset_path)
    data = ds.load_data()

    print(f"\n✓ Loaded {len(data)} samples")
    sample = data[0]
    print(f"\nSample 0:")
    print(f"  index: {sample['index']}")
    print(f"  question: {sample['question'][:100]}...")
    print(f"  answer: {sample['answer']}")
    print(f"  target_instances: {sample['target_instances']}")
    print(f"  image_path: {sample['image_path']}")

    print(f"\nFormatted prompt for sample 0:")
    print(ds.format_prompt(sample))
