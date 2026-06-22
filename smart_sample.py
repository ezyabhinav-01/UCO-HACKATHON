"""
smart_sample.py
Run this ONCE to create a balanced, speaker-diverse subset from your large datasets.
Put this anywhere, run from Anaconda Prompt.
"""

import os
import csv
import json
import random
import shutil

random.seed(42)  # makes sampling reproducible — same result every run

# ============================================================
# COMMON VOICE HINDI — uses validated.tsv for speaker info
# ============================================================

def sample_common_voice(
    cv_folder,           # folder where you extracted Common Voice Hindi
    output_folder,       # where to copy selected files
    files_per_speaker=5, # how many clips per unique speaker
    max_speakers=200     # cap total speakers (200 × 5 = 1000 files)
):
    """
    Common Voice has a file called validated.tsv
    Each row = one audio clip
    Column 'client_id' = unique speaker ID
    Column 'path' = filename (inside clips/ subfolder)
    
    This function reads that TSV, groups by speaker,
    takes files_per_speaker clips from each speaker,
    copies them to output_folder.
    """
    
    tsv_path = os.path.join(cv_folder, "validated.tsv")
    clips_folder = os.path.join(cv_folder, "clips")
    
    if not os.path.exists(tsv_path):
        print(f"ERROR: validated.tsv not found in {cv_folder}")
        print("Make sure you extracted Common Voice correctly")
        return []
    
    # Read the TSV file
    speakers = {}  # {speaker_id: [list of filenames]}
    
    with open(tsv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            spk = row['client_id']
            fname = row['path']
            if spk not in speakers:
                speakers[spk] = []
            speakers[spk].append(fname)
    
    print(f"Common Voice Hindi: {len(speakers)} unique speakers found")
    print(f"Total clips available: {sum(len(v) for v in speakers.values())}")
    
    # Shuffle speakers so we don't always pick same ones
    speaker_list = list(speakers.keys())
    random.shuffle(speaker_list)
    speaker_list = speaker_list[:max_speakers]  # cap at max_speakers
    
    os.makedirs(output_folder, exist_ok=True)
    copied_files = []
    
    for spk_id in speaker_list:
        clips = speakers[spk_id]
        random.shuffle(clips)
        selected = clips[:files_per_speaker]
        
        for fname in selected:
            # Common Voice clips are .mp3 format originally. We convert to .wav for consistency
            src = os.path.join(clips_folder, fname)
            if not os.path.exists(src):
                continue
            
            # Create safe output filename using speaker prefix
            safe_spk = spk_id[:8]  # first 8 chars of speaker ID
            wav_fname = fname.replace('.mp3', '.wav')
            out_name = f"cv_{safe_spk}_{wav_fname.replace('/', '_')}"
            dst = os.path.join(output_folder, out_name)
            
            try:
                import librosa
                import soundfile as sf
                y, sr = librosa.load(src, sr=16000)
                sf.write(dst, y, sr, format='WAV', subtype='PCM_16')
                copied_files.append({
                    'path': dst,
                    'speaker_id': f"cv_{safe_spk}",
                    'source': 'common_voice_hindi',
                    'label': 0  # real
                })
            except Exception as e:
                print(f"ERROR: Failed to convert {src} to WAV: {e}")
    
    print(f"Common Voice: copied {len(copied_files)} files from {len(speaker_list)} speakers")
    return copied_files


# ============================================================
# WAVEFAKE — no speaker metadata, but we know structure:
# real/ = LJSpeech (1 speaker — limit heavily)
# fake/ = AI generated (treat each vocoder subfolder as "different source")
# ============================================================

def sample_wavefake_real(
    ljspeech_wavs_folder,
    output_folder,
    max_files=150           # hard cap — 1 speaker, don't need many
):
    """
    LJSpeech is 1 speaker so we just randomly sample a small number.
    150 is enough — more doesn't add speaker diversity.
    """
    
    all_files = [f for f in os.listdir(ljspeech_wavs_folder)
                 if f.endswith('.wav')]
    random.shuffle(all_files)
    selected = all_files[:max_files]
    
    os.makedirs(output_folder, exist_ok=True)
    copied_files = []
    
    for fname in selected:
        src = os.path.join(ljspeech_wavs_folder, fname)
        dst = os.path.join(output_folder, fname)
        shutil.copy2(src, dst)
        copied_files.append({
            'path': dst,
            'speaker_id': 'ljspeech_linda',
            'source': 'ljspeech',
            'label': 0
        })
    
    print(f"LJSpeech real: copied {len(copied_files)} files (1 speaker — intentionally small)")
    return copied_files


def sample_wavefake_fake(
    wavefake_folder,         # folder containing hifigan/ melgan/ etc subfolders
    output_folder,
    files_per_vocoder=400    # 400 per vocoder × however many vocoders you have
):
    """
    WaveFake fake folder structure:
    wavefake/
      ljspeech_hifiGAN/        ← subfolder per vocoder
      ljspeech_full_band_melgan/
      ljspeech_multi_band_melgan/
      etc.
    
    We sample equally from each vocoder subfolder.
    This ensures the model sees artifacts from MULTIPLE
    different AI generation methods — not just one.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    copied_files = []
    
    # Find all subfolders (each = one vocoder)
    subfolders = [d for d in os.listdir(wavefake_folder)
                  if os.path.isdir(os.path.join(wavefake_folder, d))]
    
    if not subfolders:
        # Maybe files are directly in the folder, not in subfolders
        all_files = [f for f in os.listdir(wavefake_folder) if f.endswith('.wav')]
        random.shuffle(all_files)
        selected = all_files[:files_per_vocoder]
        for fname in selected:
            src = os.path.join(wavefake_folder, fname)
            dst = os.path.join(output_folder, f"wf_direct_{fname}")
            shutil.copy2(src, dst)
            copied_files.append({
                'path': dst,
                'speaker_id': 'wavefake_linda',
                'source': 'wavefake',
                'label': 1
            })
        print(f"WaveFake fake: {len(copied_files)} files (no subfolders found)")
        return copied_files
    
    print(f"WaveFake subfolders found: {subfolders}")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(wavefake_folder, subfolder)
        all_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
        random.shuffle(all_files)
        selected = all_files[:files_per_vocoder]
        
        vocoder_name = subfolder.replace('ljspeech_', '')  # cleaner name
        
        for fname in selected:
            src = os.path.join(subfolder_path, fname)
            dst = os.path.join(output_folder, f"wf_{vocoder_name}_{fname}")
            shutil.copy2(src, dst)
            copied_files.append({
                'path': dst,
                'speaker_id': f'wavefake_{vocoder_name}',
                'source': f'wavefake_{vocoder_name}',
                'label': 1
            })
        
        print(f"  {subfolder}: copied {len(selected)} files")
    
    print(f"WaveFake fake total: {len(copied_files)} files")
    return copied_files


# ============================================================
# SVARAH — extracted from HuggingFace parquet files
# ============================================================

def sample_svarah(
    svarah_folder,
    output_folder,
    files_per_speaker=4,
    max_speakers=117
):
    """
    Reads Svarah dataset from HuggingFace Parquet files,
    samples speech clips per speaker, and saves them to wav files.
    """
    import pandas as pd
    
    parquet_files = [os.path.join(svarah_folder, f) for f in os.listdir(svarah_folder) if f.endswith('.parquet')]
    if not parquet_files:
        print("ERROR: No Svarah parquet files found in", svarah_folder)
        return []
    
    print(f"Svarah: reading from {len(parquet_files)} parquet files...")
    dfs = []
    for pf in parquet_files:
        dfs.append(pd.read_parquet(pf))
    df = pd.concat(dfs, ignore_index=True)
    
    # Extract speaker_id from filename in audio_filepath metadata
    # e.g., '281474976884635_f3269_chunk_0.wav' -> use the first part as speaker ID
    def extract_speaker(row):
        path_val = row['audio_filepath']['path']
        parts = path_val.split('_')
        return parts[0] if len(parts) > 0 else "unknown"
        
    df['speaker_id'] = df.apply(extract_speaker, axis=1)
    speakers = df['speaker_id'].unique()
    print(f"Svarah: {len(speakers)} unique speakers found")
    
    speaker_list = list(speakers)
    random.shuffle(speaker_list)
    speaker_list = speaker_list[:max_speakers]
    
    os.makedirs(output_folder, exist_ok=True)
    copied_files = []
    
    for spk_id in speaker_list:
        spk_df = df[df['speaker_id'] == spk_id]
        selected_rows = spk_df.sample(n=min(len(spk_df), files_per_speaker), random_state=42)
        
        for idx, row in selected_rows.iterrows():
            audio_data = row['audio_filepath']
            filename = audio_data['path']
            audio_bytes = audio_data['bytes']
            
            out_name = f"svarah_{spk_id}_{filename}"
            dst = os.path.join(output_folder, out_name)
            
            with open(dst, 'wb') as f:
                f.write(audio_bytes)
                
            copied_files.append({
                'path': dst,
                'speaker_id': f"svarah_{spk_id}",
                'source': 'svarah',
                'label': 0
            })
            
    print(f"Svarah: extracted and saved {len(copied_files)} files from {len(speaker_list)} speakers")
    return copied_files


# ============================================================
# TEAM VOICES — already organized, just register them
# ============================================================

def register_team_voices(base_folder="e:/UCO-HACKATHON/data"):
    """
    Doesn't copy anything — just registers existing files
    with their speaker IDs into the manifest list.
    """
    registered = []
    
    for member in ["vishal", "abhinav", "aditya", "dhruv"]:
        # Real voices
        real_folder = os.path.join(base_folder, "real_voices", member)
        if os.path.exists(real_folder):
            for f in os.listdir(real_folder):
                if f.endswith('.wav'):
                    registered.append({
                        'path': os.path.join(real_folder, f),
                        'speaker_id': member,
                        'source': 'team_real',
                        'label': 0
                    })
        
        # Clone voices
        clone_folder = os.path.join(base_folder, "clones", member)
        if os.path.exists(clone_folder):
            for f in os.listdir(clone_folder):
                if f.endswith('.wav'):
                    registered.append({
                        'path': os.path.join(clone_folder, f),
                        'speaker_id': member,  # same ID as real — split together
                        'source': 'team_clone',
                        'label': 1
                    })
    
    real_count = sum(1 for r in registered if r['label'] == 0)
    fake_count = sum(1 for r in registered if r['label'] == 1)
    print(f"Team voices: {real_count} real, {fake_count} clones registered")
    return registered


# ============================================================
# SAVE FINAL MANIFEST
# ============================================================

def save_manifest(all_entries, output_path="e:/UCO-HACKATHON/data/processed/manifest.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'speaker_id', 'source', 'label'])
        writer.writeheader()
        writer.writerows(all_entries)
    
    total = len(all_entries)
    real = sum(1 for e in all_entries if e['label'] == 0)
    fake = sum(1 for e in all_entries if e['label'] == 1)
    speakers = len(set(e['speaker_id'] for e in all_entries))
    
    print("\n" + "="*50)
    print("MANIFEST SUMMARY")
    print("="*50)
    print(f"Total files    : {total}")
    print(f"Real (label=0) : {real}")
    print(f"Fake (label=1) : {fake}")
    print(f"Unique speakers: {speakers}")
    print(f"Balance ratio  : {real/total*100:.1f}% real / {fake/total*100:.1f}% fake")
    
    if abs(real - fake) / total > 0.2:
        print("\nWARNING: Dataset is imbalanced (>20% difference between real and fake)")
        print("Consider adding more data to the smaller category")
    else:
        print("\nBalance looks good!")
    
    print(f"\nManifest saved to: {output_path}")


# ============================================================
# MAIN — ADJUST PATHS TO MATCH YOUR ACTUAL FOLDER LOCATIONS
# ============================================================

if __name__ == "__main__":
    
    all_entries = []
    
    # 1. Team voices (real + clones)
    all_entries += register_team_voices(
        base_folder="e:/UCO-HACKATHON/data"
    )
    
    # 2. Common Voice Hindi
    all_entries += sample_common_voice(
        cv_folder="D:/Downloads!!/1774126233883-cv-corpus-25.0-2026-03-09-hi/cv-corpus-25.0-2026-03-09/hi",
        output_folder="e:/UCO-HACKATHON/data/sources/real/common_voice_hindi",
        files_per_speaker=5,
        max_speakers=200
    )
    
    # 3. LJSpeech real (small slice only)
    all_entries += sample_wavefake_real(
        ljspeech_wavs_folder="D:/Downloads!!/LJSpeech-1.1/LJSpeech-1.1/wavs",
        output_folder="e:/UCO-HACKATHON/data/sources/real/ljspeech",
        max_files=150
    )
    
    # 4. WaveFake fake
    all_entries += sample_wavefake_fake(
        wavefake_folder="D:/Downloads!!/sources_wavefake/_sources_wavefake/fake_voice",
        output_folder="e:/UCO-HACKATHON/data/sources/fake/wavefake_fake",
        files_per_vocoder=400
    )
    
    # 5. Svarah
    all_entries += sample_svarah(
        svarah_folder="D:/Downloads!!/SvarahReal/data",
        output_folder="e:/UCO-HACKATHON/data/sources/real/svarah",
        files_per_speaker=4,
        max_speakers=117
    )
    
    # Save final manifest
    save_manifest(all_entries)
