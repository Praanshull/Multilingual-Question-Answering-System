"""
Model Loading Module
Handles loading mBART + LoRA model from disk
"""

import torch
import gc
from pathlib import Path
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from peft import PeftModel


class ModelLoader:
    """Handles model and tokenizer loading"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize ModelLoader
        
        Args:
            model_path: Path to saved model directory
        """
        self.model_path = model_path or "models/multilingual_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model and tokenizer from disk"""
        print(f"🔧 Loading model from: {self.model_path}")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Load tokenizer
            print("⏳ Loading tokenizer...")
            self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_path)
            print("✅ Tokenizer loaded")
            
            # Load base model
            print("⏳ Loading base mBART model...")
            base_model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
            print("✅ Base model loaded")
            
            # Load LoRA weights
            print("⏳ Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            print("✅ LoRA weights loaded")
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"\n✅ MODEL LOADED SUCCESSFULLY!")
            print(f"💾 Device: {self.device}")
            print(f"📊 Total parameters: {self.model.num_parameters():,}")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"\n❌ ERROR LOADING MODEL: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
            
        return {
            "device": str(self.device),
            "parameters": self.model.num_parameters(),
            "model_path": self.model_path,
            "base_model": "facebook/mbart-large-50-many-to-many-mmt"
        }