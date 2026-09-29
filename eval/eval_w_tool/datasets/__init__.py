"""
Dataset handlers for different VQA benchmarks.
All datasets output a unified format: (question, prompt, image_path)
"""
from .base import DatasetBase
from .hrbench import HRBenchDataset
from .vstar import VStarDataset
from .fixretinasft import FixRetinaSFTDataset
from .generalvqa import GeneralVQADataset
from .mmerealworldlite import MMERealWorldLiteDataset
from .visualprobe import VisualProbeDataset
from .treebench import TreeBenchDataset
from .treevgr import TreeVGRRLDataset

__all__ = [
    'DatasetBase', 
    'HRBenchDataset', 
    'VStarDataset', 
    'FixRetinaSFTDataset', 
    'GeneralVQADataset', 
    'MMERealWorldLiteDataset',
    'VisualProbeDataset',
    'TreeBenchDataset',
    'TreeVGRRLDataset',
]

