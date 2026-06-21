import os
import sys
import argparse
import torch
import numpy as np
from scipy.ndimage import zoom
import warnings

warnings.filterwarnings("ignore")

# Import preprocessing
from preprocess import load_and_standardize, audio_to_mel_spectrogram
# Import PhaseGuard Layer 1 model definition
from train_layer1 import PhaseGuardL1

def main():
    parser = argparse.ArgumentParser(description="Test a voice file with PhaseGuard Layer 1 (AI Voice Authenticity)")
    parser.add_argument("file_path", type=str, help="Path to the WAV audio file to test")
    args = parser.parse_args()
    
    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"ERROR: Audio file not found: {file_path}")
        sys.exit(1)
        
    model_path = "models/layer1_mobilenet.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model weights not found at: {model_path}")
        print("Please run train_layer1.py first to train the model.")
        sys.exit(1)
        
    print(f"Loading PhaseGuard Layer 1 model weights from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseGuardL1()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loading and preprocessing audio: {file_path}...")
    try:
        # Load and pad/truncate to 2.0s
        audio = load_and_standardize(file_path)
        # Convert to Mel-Spectrogram
        mel = audio_to_mel_spectrogram(audio)
        
        # Standardize mel shape to (128, 128)
        if mel.shape[1] != 128:
            mel_resized = zoom(mel, (1, 128 / mel.shape[1]))
        else:
            mel_resized = mel
            
        mel_min = mel_resized.min()
        mel_max = mel_resized.max()
        mel_norm = (mel_resized - mel_min) / (mel_max - mel_min + 1e-10)
        
        # Prepare tensor (shape: 1, 1, 128, 128)
        tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(device)
    except Exception as e:
        print(f"ERROR: Preprocessing failed: {e}")
        sys.exit(1)
        
    print("Running AI Voice Authenticity inference...")
    with torch.no_grad():
        prediction = model(tensor).item()
        
    print("\n" + "="*50)
    print("                PHASEGUARD L1 RESULTS")
    print("="*50)
    print(f"Audio Tested: {os.path.basename(file_path)}")
    print(f"AI Score    : {prediction:.4f}")
    
    if prediction > 0.5:
        confidence = prediction * 100
        print(f"DECISION    : 🚨 FAKE / AI VOICE CLONE (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - prediction) * 100
        print(f"DECISION    : ✅ REAL HUMAN VOICE (Confidence: {confidence:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
