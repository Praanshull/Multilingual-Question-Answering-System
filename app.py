"""
Main Application Entry Point
Multilingual Question Answering System with Gradio Interface
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.model_loader import ModelLoader
from app.inference import QAInference
from app.interface import create_interface


def main():
    """Main application entry point"""
    
    print("=" * 80)
    print("🚀 INITIALIZING MULTILINGUAL QA SYSTEM")
    print("=" * 80)
    
    # Configuration
    MODEL_PATH = "models/multilingual_model"  # Change this to your model path
    
    # Load model
    print(f"\n📂 Model path: {MODEL_PATH}")
    loader = ModelLoader(model_path=MODEL_PATH)
    
    try:
        model, tokenizer = loader.load()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\n💡 Please ensure:")
        print(f"   1. Model exists at: {MODEL_PATH}")
        print("   2. All required files are present")
        print("   3. You have sufficient memory")
        return
    
    # Create inference engine
    print("\n🔧 Initializing inference engine...")
    inference_engine = QAInference(
        model=model,
        tokenizer=tokenizer,
        device=loader.device
    )
    print("✅ Inference engine ready")
    
    # Create interface
    print("\n🎨 Building Gradio interface...")
    demo = create_interface(inference_engine)
    print("✅ Interface created")
    
    # Launch
    print("\n" + "=" * 80)
    print("🚀 LAUNCHING APPLICATION")
    print("=" * 80)

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #3498db, #e74c3c);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True for public URL
        show_error=True,
        quiet=False,
        css=custom_css
    )
    
    print("\n✅ Application launched successfully!")
    print("📱 Access the interface at: http://localhost:7860")
    print("\n💡 TIP: Set share=True in demo.launch() to get a public URL")
    print("=" * 80)


if __name__ == "__main__":
    main()