
import os
import sys

# Add the parent directory to sys.path so we can import from app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.agent import get_vectorstore
    print("Attempting to call get_vectorstore()...")
    vectorstore = get_vectorstore()
    print("Successfully called get_vectorstore()")
except AttributeError as e:
    print(f"Caught expected AttributeError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {e}")
