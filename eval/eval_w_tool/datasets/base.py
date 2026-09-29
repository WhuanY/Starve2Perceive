"""
Base class for dataset handlers.
All datasets should inherit from this and implement the required methods.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import os


class DatasetBase(ABC):
    """Base class for all dataset handlers."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset handler.
        
        Args:
            dataset_path: Path to the dataset file or directory
        """
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    @abstractmethod
    def load_data(self, start_index: int = 0, end_index: int = -1) -> List[Dict[str, Any]]:
        """
        Load the dataset from the given path. 
        
        Returns:
            List of data samples, each containing at least:
            - question: str
            - prompt: str (formatted with question and options)
            - image_path: str or List[str]
            - answer: str (ground truth answer)
            - index: Any (unique identifier)
        """
        pass
    
    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a prompt from a sample for the model.
        
        Args:
            sample: A single data sample
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def __iter__(self):
        """Make the dataset iterable."""
        return iter(self.load_data())
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.load_data())

