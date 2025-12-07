# 🌍 Multilingual Question Answering System

A state-of-the-art multilingual question answering system supporting **English 🇬🇧** and **German 🇩🇪**, built with **mBART-large-50** fine-tuned using **LoRA** (Low-Rank Adaptation).

![Model](https://img.shields.io/badge/Model-mBART--large--50-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- Deployement
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Training](#training)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project implements a **bilingual extractive question answering system** that can:
- Extract answers from English contexts
- Extract answers from German contexts
- Achieve **high accuracy** with minimal training data through transfer learning
- Run efficiently using **Parameter-Efficient Fine-Tuning (LoRA)**

### What is Extractive QA?
The model reads a passage (context) and a question, then extracts the exact answer span from the context.

**Example:**
- **Question:** "What is the capital of France?"
- **Context:** "Paris is the capital and most populous city of France."
- **Answer:** "Paris"

---
## 🎯 Deployement
-**Deployed on:** https://huggingface.co/spaces/Praanshull/multilingual-qa-system

## ✨ Key Features

✅ **Bilingual Support** - English and German
✅ **Fast Inference** - <1 second per query on GPU
✅ **Memory Efficient** - Uses LoRA (only 0.29% trainable parameters)
✅ **High Accuracy** - >65% F1 score on both languages
✅ **Easy Deployment** - Gradio web interface included
✅ **Well Documented** - Comprehensive code comments and README

---

## 📊 Performance

### Model Metrics

| Metric | English (SQuAD) | German (XQuAD) | Improvement |
|--------|----------------|----------------|-------------|
| **BLEU** | 37.79 | **43.12** | +5.33 |
| **ROUGE-L** | 0.6272 | **0.6622** | +0.035 |
| **Exact Match** | 43.60% | **48.74%** | +5.14% |
| **F1 Score** | 0.6329 | **0.6580** | +0.025 |
| **Avg (EM+F1)** | 0.5344 | **0.5727** | +0.038 |

### Key Insights
- 🎉 **German achieves 107.2% of English performance** despite having only ~5% of training data
- 🚀 Strong **transfer learning** from English to German
- 💪 Better German scores demonstrate effective **cross-lingual adaptation**

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Praanshull/multilingual-qa-system.git
cd multilingual-qa-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the model**
```bash
# Option 1: Download from your Google Drive
# (Replace with your actual model path)

# Option 2: Use Hugging Face (if uploaded)
# Will be automatically downloaded on first run
```

---

## 📁 Project Structure

```
Multilingual-QA-System/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── model_loader.py       # Model loading logic
│   ├── inference.py          # Inference/prediction engine
│   ├── interface.py          # Gradio UI components
│   └── utils.py              # Utility functions
│
├── models/
│   └── multilingual_model/   # Saved model files
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       ├── tokenizer_config.json
│       └── ...
│
├── checkpoints/              # Training checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
│
├── logs/                     # Training logs
│   └── training.log
│
├── notebook/                 # Original Jupyter notebook
│   └── main.ipynb
│
├── app.py                    # Main application entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
└── LICENSE                   # MIT License

```

---

## 💻 Usage

### 1. Launch the Web Interface

```bash
python app.py
```

Then open your browser to **http://localhost:7860**

### 2. Programmatic Usage

```python
from app.model_loader import ModelLoader
from app.inference import QAInference

# Load model
loader = ModelLoader(model_path="models/multilingual_model")
model, tokenizer = loader.load()

# Create inference engine
qa = QAInference(model, tokenizer, loader.device)

# English example
answer, info = qa.answer_question(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France.",
    language="English"
)
print(f"Answer: {answer}")

# German example
answer_de, info_de = qa.answer_question(
    question="Was ist die Hauptstadt von Deutschland?",
    context="Berlin ist die Hauptstadt von Deutschland.",
    language="German"
)
print(f"Antwort: {answer_de}")
```

### 3. API Server (Coming Soon)

```bash
# Launch FastAPI server
python -m app.api --host 0.0.0.0 --port 8000
```

---

## 🧠 Model Details

### Architecture
- **Base Model:** `facebook/mbart-large-50-many-to-many-mmt`
  - 610M total parameters
  - Pre-trained on 50 languages
  - Sequence-to-sequence architecture

- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
  - Rank (r): 8
  - Alpha: 32
  - Target modules: `q_proj`, `k_proj`, `v_proj`
  - Only **1.77M trainable parameters** (0.29% of total)

### Training Data
- **English:** SQuAD v1.1
  - 20,000 samples (from 87,599 available)
  - Balanced sampling across topics
  
- **German:** XQuAD (German)
  - ~950 samples (80% of 1,190 available)
  - Cross-lingual evaluation dataset

### Hyperparameters
```python
{
    "learning_rate": 3e-4,
    "batch_size": 16 (2 * 8 gradient accumulation),
    "epochs": 3,
    "max_source_length": 256,
    "max_target_length": 64,
    "fp16": True,
    "optimizer": "AdamW",
    "weight_decay": 0.01
}
```

---

## 🔧 Training

### Train from Scratch

```bash
# See notebook/main.ipynb for full training pipeline
jupyter notebook notebook/main.ipynb
```

### Key Training Steps

1. **Data Preparation**
   - Load SQuAD and XQuAD datasets
   - Convert to text-to-text format
   - Tokenize with mBART tokenizer

2. **Model Setup**
   - Load base mBART model
   - Apply LoRA configuration
   - Configure language tokens

3. **Training**
   - English: 3 epochs (~2 hours on T4 GPU)
   - German: 3 epochs (~30 minutes on T4 GPU)
   - Total: ~2.5 hours

4. **Evaluation**
   - BLEU, ROUGE, Exact Match, F1
   - Cross-lingual performance analysis

---

## ⚠️ Limitations

### Current Constraints
1. **Long Context** - Performance degrades with passages >500 words
2. **Complex Questions** - Multi-hop reasoning not supported
3. **Answer Presence** - Answer must be explicitly stated in context
4. **Languages** - Only English and German supported
5. **Training Data** - Limited to 20K English + 1K German samples

### Why These Exist
- ✂️ **Context truncation** due to GPU memory constraints
- 🧮 **Simple architecture** optimized for extractive QA only
- ⚡ **Fast training** prioritized over maximum performance

---

## 🎯 Future Improvements

- [ ] Increase context window to 512 tokens
- [ ] Add more languages (French, Spanish, Chinese)
- [ ] Implement answer confidence scoring
- [ ] Add data augmentation techniques
- [ ] Deploy as REST API with FastAPI
- [ ] Create Docker container for easy deployment
- [ ] Add answer verification layer
- [ ] Support generative (non-extractive) answers

---

## 📖 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{verma2025multilingual_qa,
  author = {Verma, Praanshull},
  title = {Multilingual Question Answering System with mBART and LoRA},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Praanshull/multilingual-qa-system}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Praanshull Verma**
- GitHub: [@Praanshull](https://github.com/Praanshull)
- LinkedIn: [Your LinkedIn]

---

## 🙏 Acknowledgments

- **Hugging Face** - For Transformers library and model hosting
- **Facebook AI** - For mBART pre-trained model
- **Stanford NLP** - For SQuAD dataset
- **Google Research** - For XQuAD dataset
- **PEFT Team** - For LoRA implementation

---

## 📞 Support

If you encounter any issues or have questions:

1. Check [Issues](https://github.com/Praanshull/multilingual-qa-system/issues)
2. Create a new issue with detailed description
3. Reach out on LinkedIn

---

<div align="center">

**Built with ❤️ using PyTorch, Transformers, and Gradio**

⭐ Star this repo if you find it helpful!

</div>
