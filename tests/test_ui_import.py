
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Testing imports...")
    from src.inference import InferenceEngine
    from src.ui_styles import get_main_css
    from src.monitoring import monitor
    print("src imports successful.")
    
    import streamlit
    print("streamlit import successful.")
    
    print("All checks passed.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
