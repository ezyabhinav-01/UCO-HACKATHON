"""
process_phase1.py  (v2 — multi-source, speaker-aware)

Folder structure expected:
  data/sources/ljspeech/wavs/*.wav            (1 speaker: "ljspeech")
  data/sources/wavefake_fake/*.wav            (label=fake, speaker="wavefake")
  data/sources/common_voice_hindi/*.wav       (many speakers — use filename or folder as ID)
  data/sources/svarah/*.wav
  data/sources/asvspoof_fake/*.wav
  data/raw_voices/<member>/*.wav              (label=real, speaker=<member>)
  data/clones/<member>/*.wav                  (label=fake, speaker=<member>)
"""

import os
import csv
import random
import numpy as np
import librosa
from scipy.ndimage import zoom

# ============================================================
# STEP 1: BUILD THE MANIFEST
# A manifest is just a table: filepath | label | speaker_id | source
# ============================================================

def build_manifest():
    rows = []  # each row = [filepath, label(0/1), speaker_id, source_name]
    
    # ---- Team real voices: each member is their own speaker ----
    for member in ["vishal", "abhinav", "aditya", "dhruv"]:
        folder = f"data/real_voices/{member}"
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith('.wav'):
                    rows.append([os.path.join(folder, f), 0, member, "team_real"])
    
    # ---- Team clones: each member's clone = same speaker_id as their real voice ----
    # (so train/test split keeps a person's real+fake together, avoiding leakage)
    for member in ["vishal", "abhinav", "aditya", "dhruv"]:
        folder = f"data/clones/{member}"
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith('.wav'):
                    rows.append([os.path.join(folder, f), 1, member, "team_clone"])
    
    # ---- LJSpeech (1 speaker, capped to 200 files) ----
    folder = "data/sources/ljspeech/wavs"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        random.shuffle(files)
        for f in files[:200]:
            rows.append([os.path.join(folder, f), 0, "ljspeech_linda", "ljspeech"])
    
    # ---- WaveFake generated fakes (cap to 500, all treated as speaker "ljspeech_linda" too,
    #      since they're vocoder versions of the same voice) ----
    folder = "data/sources/wavefake_fake"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        random.shuffle(files)
        for f in files[:500]:
            rows.append([os.path.join(folder, f), 1, "ljspeech_linda", "wavefake"])
    
    # ---- Common Voice Hindi (many speakers — use first part of filename as speaker proxy) ----
    folder = "data/sources/common_voice_hindi"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        random.shuffle(files)
        for f in files[:800]:
            # common_voice filenames usually have format: common_voice_hi_XXXXXX.wav or similar.
            # Split by underscores or take first part
            parts = f.split('_')
            speaker_id = parts[-1].split('.')[0] if len(parts) > 1 else f.split('.')[0]
            rows.append([os.path.join(folder, f), 0, f"cv_{speaker_id}", "common_voice"])
    
    # ---- Svarah (Indian English, many speakers) ----
    folder = "data/sources/svarah"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        random.shuffle(files)
        for f in files[:500]:
            speaker_id = f.split('_')[0]
            rows.append([os.path.join(folder, f), 0, f"svarah_{speaker_id}", "svarah"])
    
    # ---- ASVspoof fakes (many speakers from VCTK) ----
    folder = "data/sources/asvspoof_fake"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        random.shuffle(files)
        for f in files[:800]:
            speaker_id = f.split('_')[0]
            rows.append([os.path.join(folder, f), 1, f"asv_{speaker_id}", "asvspoof"])
    
    # Save manifest as CSV
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/manifest.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "speaker_id", "source"])
        writer.writerows(rows)
    
    print(f"Manifest built: {len(rows)} total files")
    print(f"  Real (0): {sum(1 for r in rows if r[1]==0)}")
    print(f"  Fake (1): {sum(1 for r in rows if r[1]==1)}")
    print(f"  Unique speakers: {len(set(r[2] for r in rows))}")
    
    return rows


# ============================================================
# STEP 2: SPEAKER-AWARE TRAIN/TEST SPLIT
# Whole speakers go to train OR test, never split across both
# ============================================================

def speaker_split(rows, test_fraction=0.2):
    # Get unique speakers
    speakers = list(set(r[2] for r in rows))
    random.shuffle(speakers)
    
    split_point = int(len(speakers) * (1 - test_fraction))
    train_speakers = set(speakers[:split_point])
    test_speakers = set(speakers[split_point:])
    
    train_rows = [r for r in rows if r[2] in train_speakers]
    test_rows = [r for r in rows if r[2] in test_speakers]
    
    print(f"\nTrain: {len(train_rows)} files from {len(train_speakers)} speakers")
    print(f"Test:  {len(test_rows)} files from {len(test_speakers)} speakers")
    
    return train_rows, test_rows


# ============================================================
# STEP 3: AUDIO PROCESSING
# ============================================================

def load_and_standardize(file_path, target_sr=16000, duration=2.0):
    try:
        import soundfile as sf
        # Use soundfile directly to prevent Windows codec errors
        y, sr = sf.read(file_path)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
    except Exception:
        return None
    target_length = int(target_sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    y = y[:target_length]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y

def audio_to_mel_spectrogram(y, sr=16000):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=512, hop_length=160)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = zoom(mel_db, (1.0, 128.0 / mel_db.shape[1]))
    mn, mx = mel_resized.min(), mel_resized.max()
    return (mel_resized - mn) / (mx - mn + 1e-10) if mx > mn else np.zeros_like(mel_resized)

def extract_5_signals(y, sr=16000):
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    phase_diff = np.diff(np.angle(stft), axis=1)
    phase_jump_rate = np.sum(np.abs(phase_diff) > np.pi*0.5) / phase_diff.size
    
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        jitter = float(np.mean(np.abs(np.diff(f0_clean))) / (np.mean(f0_clean)+1e-10)) if len(f0_clean) > 2 else 0.0
    except Exception:
        jitter = 0.0
        
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    noise_floor = float(np.percentile(rms, 10))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta_var = float(np.var(librosa.feature.delta(mfcc)))
    
    return [float(phase_jump_rate), jitter, flatness, noise_floor, mfcc_delta_var]


# ============================================================
# STEP 4: PROCESS A SET OF ROWS INTO ARRAYS
# ============================================================

def process_rows(rows, name):
    specs, feats, labels = [], [], []
    for i, (filepath, label, speaker, source) in enumerate(rows):
        if (i+1) % 100 == 0 or (i+1) == len(rows):
            print(f"  [{name}] processed {i+1}/{len(rows)} files...")
        audio = load_and_standardize(filepath)
        if audio is None:
            continue
        specs.append(audio_to_mel_spectrogram(audio))
        feats.append(extract_5_signals(audio))
        labels.append(label)
    return np.array(specs, dtype=np.float32), np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    rows = build_manifest()
    if len(rows) > 0:
        train_rows, test_rows = speaker_split(rows)
        
        print("\nProcessing TRAIN set...")
        train_specs, train_feats, train_labels = process_rows(train_rows, "train")
        
        print("\nProcessing TEST set...")
        test_specs, test_feats, test_labels = process_rows(test_rows, "test")
        
        np.save("data/processed/train_specs.npy", train_specs)
        np.save("data/processed/train_feats.npy", train_feats)
        np.save("data/processed/train_labels.npy", train_labels)
        np.save("data/processed/test_specs.npy", test_specs)
        np.save("data/processed/test_feats.npy", test_feats)
        np.save("data/processed/test_labels.npy", test_labels)
        
        print("\n[OK] Done. Train/test sets saved separately, split by speaker.")
    else:
        print("No files found. Populate data folders first.")
