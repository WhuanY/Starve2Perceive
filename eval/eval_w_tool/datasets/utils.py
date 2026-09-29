import json
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load jsonl file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load json file."""
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # testing
    print("hello")
    data = load_jsonl("/home/ywuit/vlmpaper/data/FixRetina_SFT/sft/fix_retina_sft_qa.jsonl")
    print("data: ", data)
    print("len(data): ", len(data))