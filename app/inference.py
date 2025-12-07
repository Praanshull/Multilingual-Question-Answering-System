"""
Inference Module
Handles question answering predictions
"""

import torch
from typing import Tuple


class QAInference:
    """Handles question answering inference"""
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize QA Inference
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            device: Torch device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def answer_question(
        self, 
        question: str, 
        context: str, 
        language: str = "English",
        max_length: int = 64
    ) -> Tuple[str, str]:
        """
        Generate answer for given question and context
        
        Args:
            question: Question text
            context: Context/passage text
            language: "English" or "German"
            max_length: Maximum answer length
            
        Returns:
            Tuple of (answer, response_info)
        """
        if not question.strip() or not context.strip():
            return "⚠️ Please provide both a question and context!", ""
        
        try:
            # Configure language
            if language == "English":
                self.tokenizer.src_lang = "en_XX"
                self.tokenizer.tgt_lang = "en_XX"
                lang_code = self.tokenizer.lang_code_to_id["en_XX"]
            else:
                self.tokenizer.src_lang = "de_DE"
                self.tokenizer.tgt_lang = "de_DE"
                lang_code = self.tokenizer.lang_code_to_id["de_DE"]
            
            self.model.config.forced_bos_token_id = lang_code
            
            # Prepare input
            input_text = f"question: {question} context: {context}"
            inputs = self.tokenizer(
                input_text,
                max_length=256,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    forced_bos_token_id=lang_code
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence
            confidence = self._calculate_confidence(answer, context)
            
            # Format response info
            response_info = f"""
### 📊 Response Details
- **Language**: {language}
- **Answer Length**: {len(answer.split())} words
- **Confidence**: {confidence}
- **Model**: mBART-large-50 + LoRA
            """
            
            return answer, response_info
            
        except Exception as e:
            return f"❌ Error: {str(e)}", ""
    
    def _calculate_confidence(self, answer: str, context: str) -> str:
        """
        Calculate answer confidence (simple heuristic)
        
        Args:
            answer: Generated answer
            context: Input context
            
        Returns:
            Confidence level string
        """
        if len(answer.split()) < 2:
            return "Low"
        elif answer.lower() in context.lower():
            return "High"
        else:
            return "Medium"