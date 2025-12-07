"""
Multilingual Question Answering System
App package initialization
"""

__version__ = "1.0.0"
__author__ = "Praanshull Verma"

from .model_loader import ModelLoader
from .inference import QAInference
from .utils import calculate_confidence, format_answer

__all__ = [
    "ModelLoader",
    "QAInference", 
    "calculate_confidence",
    "format_answer"
]