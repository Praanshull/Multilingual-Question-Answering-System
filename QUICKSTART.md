# 🚀 Quick Start Guide

Get the Multilingual QA System up and running in **5 minutes**!

---

## ⚡ Fast Track

```bash
# 1. Clone and enter directory
git clone https://github.com/Praanshull/multilingual-qa-system.git
cd multilingual-qa-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run setup script (first time only)
python setup_project.py

# 4. Launch application
python app.py
```

Then open **http://localhost:7860** in your browser!

---

## 📋 Detailed Steps

### Step 1: Prerequisites

Make sure you have:
- ✅ Python 3.8 or higher
- ✅ pip (Python package manager)
- ✅ Git
- ✅ (Optional) CUDA-capable GPU

Check your Python version:
```bash
python --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/Praanshull/multilingual-qa-system.git
cd multilingual-qa-system
```

### Step 3: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch
- Transformers
- Gradio
- PEFT
- And other required packages

**Estimated time:** 2-5 minutes

### Step 5: Setup Project Structure

```bash
python setup_project.py
```

This script will:
1. Create necessary directories
2. Move model files to correct locations
3. Create configuration files
4. Verify everything is set up correctly

**Note:** If you haven't downloaded the model yet, you'll need to:
- Download from Google Drive (if shared)
- Or the model will be downloaded automatically on first run

### Step 6: Test the Model (Optional)

```bash
python test_model.py
```

This runs quick tests to verify everything works.

### Step 7: Launch the Application

```bash
python app.py
```

You should see:
```
================================================================================
🚀 LAUNCHING APPLICATION
================================================================================
✅ Application launched successfully!
📱 Access the interface at: http://localhost:7860
```

### Step 8: Open in Browser

Open your web browser and go to:
```
http://localhost:7860
```

---

## 🎯 Using the Interface

### Ask Questions Tab

1. **Select Language:** Choose English 🇬🇧 or German 🇩🇪
2. **Enter Question:** Type your question
3. **Provide Context:** Paste the passage containing the answer
4. **Click "Get Answer":** The model will extract the answer

**Tips:**
- Keep context under 300 words for best results
- Make sure the answer is explicitly stated in the context
- Use clear, direct questions

### Try Examples

1. Click on "Try Examples" section
2. Select example type (General Knowledge, Historical, Scientific)
3. Click "Load Example"
4. The question and context will be filled automatically
5. Click "Get Answer"

---

## 🔧 Troubleshooting

### Model Not Found Error

**Problem:** `❌ Failed to load model: Model not found`

**Solution:**
```bash
# Update the model path in app.py
MODEL_PATH = "models/multilingual_model"

# Or download the model:
python download_model.py
```

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# The model will automatically fall back to CPU
# Or reduce batch size in config if running inference in batches
```

### Port Already in Use

**Problem:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Use a different port
python app.py --port 7861
```

Or kill the process using port 7860:
```bash
# Mac/Linux
lsof -ti:7860 | xargs kill -9

# Windows
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 🌐 Deploy to Cloud

### Deploy to Hugging Face Spaces (Free)

```bash
# Install Gradio
pip install gradio

# Deploy (from project directory)
gradio deploy
```

### Deploy to Railway/Render

1. Create account on Railway/Render
2. Connect your GitHub repository
3. Set start command: `python app.py`
4. Deploy!

---

## 📚 Next Steps

Now that you have the app running:

1. ✅ Read the full [README.md](README.md) for detailed documentation
2. ✅ Check out the [notebook/main.ipynb](notebook/main.ipynb) to see training process
3. ✅ Explore the code in `app/` directory
4. ✅ Try modifying examples in `app/utils.py`
5. ✅ Add your own test cases in `test_model.py`

---

## 💡 Pro Tips

### For Development

```bash
# Enable debug mode
python app.py --debug

# Share publicly (generates public URL)
python app.py --share

# Run on specific port
python app.py --port 8080
```

### For Production

```bash
# Use gunicorn for better performance
gunicorn app:app --workers 4 --bind 0.0.0.0:7860
```

---

## ❓ Need Help?

- 📖 Check [README.md](README.md) for detailed docs
- 🐛 Report issues on [GitHub Issues](https://github.com/Praanshull/multilingual-qa-system/issues)
- 💬 Ask questions in Discussions

---

<div align="center">

**Happy Question Answering! 🎉**

[⬆️ Back to Top](#-quick-start-guide)

</div>