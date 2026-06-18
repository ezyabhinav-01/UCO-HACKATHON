import os
import sys
import subprocess

def main():
    # Detect the Streamlit binary in the current active python environment
    python_dir = os.path.dirname(sys.executable)
    
    # On Windows, binaries are in the Scripts/ subdirectory
    streamlit_bin = os.path.join(python_dir, "Scripts", "streamlit.exe")
    
    # Fallback to general command if not found
    if not os.path.exists(streamlit_bin):
        streamlit_bin = os.path.join(python_dir, "streamlit")
        if not os.path.exists(streamlit_bin):
            streamlit_bin = "streamlit"
            
    print(f"PhaseGuard Launcher: Initiating Streamlit dashboard...")
    print(f"Streamlit path: {streamlit_bin}")
    print(f"Python path: {sys.executable}\n")
    
    try:
        subprocess.run([streamlit_bin, "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\nPhaseGuard Dashboard stopped by user.")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")

if __name__ == "__main__":
    main()
