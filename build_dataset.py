import os
import numpy as np
from scipy.ndimage import zoom
from preprocess import load_and_standardize, audio_to_mel_spectrogram, extract_5_signals

def parse_asvspoof_protocol(protocol_file):
    """
    Parses ASVspoof Logical Access (LA) protocol text files.
    Format: [Speaker_ID] [File_Name] [-] [System_ID] [bonafide/spoof]
    Returns a dictionary mapping file name to label (0 = bonafide, 1 = spoof)
    """
    label_map = {}
    if not os.path.exists(protocol_file):
        print(f"ASVspoof Protocol file not found: {protocol_file}")
        return label_map
        
    print(f"Parsing ASVspoof protocol: {protocol_file}...")
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_name = parts[1]
                key = parts[4]
                # 0 = bonafide (genuine), 1 = spoof (AI/clone)
                label_map[file_name] = 0 if key == 'bonafide' else 1
    return label_map

def build_training_dataset(real_folders, fake_folders, output_path, asvspoof_folders=None):
    """
    Reads audio from standard folders (recursively) and optional ASVspoof folders.
    Extracts features and mel-spectrograms for each audio clip and saves:
    - features.npy: (N, 5) array of physical features
    - spectrograms.npy: (N, 128, 128) array of normalized mel-spectrogram images
    - labels.npy: (N,) labels (0 = Real, 1 = Fake)
    """
    all_spectrograms = []
    all_features = []
    all_labels = []
    
    # helper function to process a single file
    def process_file(filepath, label):
        try:
            audio = load_and_standardize(filepath)
            mel = audio_to_mel_spectrogram(audio)
            
            if mel.shape[1] != 128:
                mel_resized = zoom(mel, (1, 128 / mel.shape[1]))
            else:
                mel_resized = mel
                
            mel_min = mel_resized.min()
            mel_max = mel_resized.max()
            mel_norm = (mel_resized - mel_min) / (mel_max - mel_min + 1e-10)
            
            features = extract_5_signals(audio)
            feature_vector = [
                features['phase_jump_rate'],
                features['jitter'],
                features['spectral_flatness'],
                features['noise_floor'],
                features['mfcc_delta_var']
            ]
            
            all_spectrograms.append(mel_norm)
            all_features.append(feature_vector)
            all_labels.append(label)
            print(f"  [OK] {os.path.basename(filepath)} (Label: {label})")
        except Exception as e:
            print(f"  [FAIL] {os.path.basename(filepath)}: {e}")

    # 1. Process standard REAL directories recursively (Label = 0)
    print("Processing REAL/HUMAN directories recursively...")
    for folder in real_folders:
        if not os.path.exists(folder):
            print(f"Skipping missing real folder: {folder}")
            continue
        print(f"Scanning: {folder}")
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if filename.lower().endswith(('.wav', '.flac', '.mp3')):
                    filepath = os.path.join(root, filename)
                    process_file(filepath, 0)
                    
    # 2. Process standard FAKE/AI directories recursively (Label = 1)
    print("\nProcessing FAKE/AI directories recursively...")
    for folder in fake_folders:
        if not os.path.exists(folder):
            print(f"Skipping missing fake folder: {folder}")
            continue
        print(f"Scanning: {folder}")
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if filename.lower().endswith(('.wav', '.flac', '.mp3')):
                    filepath = os.path.join(root, filename)
                    process_file(filepath, 1)

    # 3. Process ASVspoof datasets (if provided)
    if asvspoof_folders:
        print("\nProcessing ASVspoof logical access datasets...")
        for asv_dir in asvspoof_folders:
            # Look for a protocol file (*.txt) inside the directory
            protocol_file = None
            audio_dir = None
            
            # Find txt protocol and wav directory
            for item in os.listdir(asv_dir):
                item_path = os.path.join(asv_dir, item)
                if os.path.isdir(item_path) and item.lower() in ['wav', 'flac', 'audio']:
                    audio_dir = item_path
                elif item.endswith('.txt') and 'protocol' in item.lower():
                    protocol_file = item_path
            
            # Defaults if not specifically named
            if not audio_dir:
                audio_dir = os.path.join(asv_dir, 'wav')
            if not protocol_file:
                # search for any txt
                txt_files = [f for f in os.listdir(asv_dir) if f.endswith('.txt')]
                if txt_files:
                    protocol_file = os.path.join(asv_dir, txt_files[0])
            
            if not os.path.exists(audio_dir):
                print(f"ASVspoof audio subdirectory not found: {audio_dir}. Skipping.")
                continue
                
            label_map = parse_asvspoof_protocol(protocol_file)
            if not label_map:
                print(f"Could not map protocols for {asv_dir}. Skipping.")
                continue
                
            print(f"Processing ASVspoof audio files in: {audio_dir}")
            for filename in os.listdir(audio_dir):
                name_without_ext = os.path.splitext(filename)[0]
                if name_without_ext in label_map:
                    filepath = os.path.join(audio_dir, filename)
                    label = label_map[name_without_ext]
                    process_file(filepath, label)

    # Save outputs
    if len(all_labels) == 0:
        print("\nERROR: No audio samples found to compile dataset!")
        return
        
    spectrograms = np.array(all_spectrograms, dtype=np.float32)
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "spectrograms.npy"), spectrograms)
    np.save(os.path.join(output_path, "features.npy"), features)
    np.save(os.path.join(output_path, "labels.npy"), labels)
    
    print(f"\nDataset Compilation Successful!")
    print(f"Total compiled samples: {len(labels)}")
    print(f"Real (0): {np.sum(labels == 0)}")
    print(f"Fake (1): {np.sum(labels == 1)}")
    print(f"Spectrogram shape: {spectrograms.shape}")

if __name__ == "__main__":
    # Folders generated by synthetic data script or recorded by user
    real_folders = [
        "data/real_voices/vishal",
        "data/real_voices/abhinav",
        "data/real_voices/aditya",
        "data/real_voices/dhruv",
        # Add Common Voice Hindi directories here:
        # "data/common_voice_hindi"
    ]
    
    fake_folders = [
        "data/clones/vishal",
        "data/clones/abhinav",
        "data/clones/aditya",
        "data/clones/dhruv",
        # Add WaveFake vocoded directories here:
        # "data/wavefake/ljspeech_full_band_melgan"
    ]
    
    # Add ASVspoof LA directory here:
    # asvspoof_folders = ["data/asvspoof"]
    
    build_training_dataset(real_folders, fake_folders, "data/processed")
