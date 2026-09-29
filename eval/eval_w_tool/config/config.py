"""
Configuration management for the VQA evaluation system.
Handles API keys, model settings, and output paths.
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for evaluation system."""
    
    # API Configuration
    api_key: str = "EMPTY"
    api_url: str = "http://localhost:8000/v1"
    eval_model_name: Optional[str] = None
    
    # Model Configuration
    model_name: str = "qwen"  # Used for saving results
    
    # Dataset Configuration
    dataset_name: str = "HRBench4K"
    dataset_path: Optional[str] = None  # Path to dataset file
    image_dir: Optional[str] = None  # Path to image directory
    
    # Output Configuration
    save_path: Optional[str] = None  # Base path for saving results
    
    # Inference Configuration
    num_workers: int = 8
    max_tokens: int = 10240
    temperature: float = 0.0
    sample_n: Optional[int] = None  # Number of samples per question

    pixel_budget_per_image: Optional[int] = None


    # Image Processing Configuration
    image_factor: int = 28
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 16384 * 28 * 28
    
    
    def get_output_dir(self) -> str:
        """Get the output directory for this model and dataset."""
        if self.save_path is None:
            raise ValueError("save_path must be set in config")
        print(self.save_path, "\n", self.model_name, "\n", self.dataset_name)
        output_dir = os.path.join(self.save_path, self.model_name, self.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_inference_output_path(self) -> str:
        """Get the path for inference results JSONL file."""
        output_dir = self.get_output_dir()
        filename = f"{self.dataset_name}_{self.model_name}_n{self.sample_n}_temperature{self.temperature}.jsonl"
        return os.path.join(output_dir, filename)
    
    def get_evaluation_output_path(self) -> str:
        """Get the path for evaluation results."""
        output_dir = self.get_output_dir()
        filename = f"{self.dataset_name}_{self.model_name}_evaluation.json"
        return os.path.join(output_dir, filename)


def load_config(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
    image_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    save_path: Optional[str] = None,
    eval_model_name: Optional[str] = None,
    num_workers: Optional[int] = None,
    sample_n: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> Config:
    """
    Load configuration from environment variables or arguments.
    
    Priority: arguments > environment variables > defaults
    
    Environment variables:
    - OPENAI_API_KEY: API key for OpenAI-compatible API
    - OPENAI_API_BASE: Base URL for API
    - MODEL_NAME: Model name for result saving
    - DATASET_NAME: Name of the dataset
    - DATASET_PATH: Path to dataset file
    - SAVE_PATH: Base path for saving results
    - EVAL_MODEL_NAME: Model name for evaluation
    - NUM_WORKERS: Number of worker processes
    """
    return Config(
        api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        api_url=api_url or os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"),
        model_name=model_name,
        image_dir=image_dir,
        dataset_name=dataset_name or os.getenv("DATASET_NAME", "HRBench4K"),
        dataset_path=dataset_path or os.getenv("DATASET_PATH"),
        save_path=save_path or os.getenv("SAVE_PATH"),
        eval_model_name=eval_model_name or os.getenv("EVAL_MODEL_NAME","qwen"),
        num_workers=num_workers or int(os.getenv("NUM_WORKERS", "8")),
        sample_n=sample_n if sample_n is not None else 1,  # Default to 1 if not provided
        temperature=temperature if temperature is not None else 0.0,  # Default to 0.0
        **kwargs
    )

