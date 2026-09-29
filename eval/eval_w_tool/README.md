# VQA Evaluation with Tool Calling

This repository provides a modular framework for evaluating Vision-Language Models (VLMs) on VQA datasets with image processing tool calling support.

## Structure

```
eval_w_tool/
├── config/              # Configuration management
│   ├── __init__.py
│   └── config.py       # Config class and loading
├── datasets/            # Dataset handlers
│   ├── __init__.py
│   ├── base.py         # Base dataset class
│   ├── hrbench.py      # HRBench dataset handler
│   └── vstar.py        # VStar dataset handler
├── inference_with_tool/ # Core inference logic
│   ├── __init__.py
│   ├── inference.py    # Main inference engine
│   ├── image_utils.py  # Image processing utilities
│   └── prompts.py      # Prompt templates
├── evaluation/          # Evaluation metrics
│   ├── __init__.py
│   └── evaluator.py    # Evaluation logic
├── run_inference.py     # Main inference script
└── run_evaluation.py    # Main evaluation script
```

## Usage

### Running Inference

```bash
python run_inference.py \
    --dataset_path /home/ywuit/vlmpaper/data/hrbench/hr_bench_4k.tsv \
    --dataset_name HRBench4K \
    --save_path /path/to/output \
    --model_name my_model \
    --api_key YOUR_API_KEY \
    --api_url http://localhost:8000/v1
```

### Running Evaluation

```bash
python run_evaluation.py \
    --results_path /path/to/inference_results.jsonl \
    --dataset_name HRBench4K \
    --output_path /path/to/evaluation_results.json
```

## Configuration

Configuration can be provided via:
1. Command line arguments
2. Environment variables (e.g., `OPENAI_API_KEY`, `OPENAI_API_BASE`)
3. Default values

## Dataset Support

Currently supported datasets:
- **HRBench4K/HRBench8K**: High-resolution benchmark in TSV format
- **VStar**: VStar benchmark in JSON format

All datasets output a unified format:
- `question`: Question text
- `prompt`: Formatted prompt with question and options
- `image_path`: Path to image file or base64 encoded image
- `answer`: Ground truth answer

## Features

- **Tool Calling**: Supports image zoom tool for detailed region inspection
- **Modular Design**: Easy to add new datasets or evaluation metrics
- **Configurable**: All settings can be customized via config or arguments
- **Unified Output Format**: All datasets produce the same output format for easy evaluation

