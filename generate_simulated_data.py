import os
import wave
import numpy as np

def generate_voice_signal(is_clone, base_f0, duration=2.0, sr=16000):
    """
    Generates a synthetic speech wave with specific physical properties:
    - Real voices: organic pitch jitter (muscle tremor), smooth phase, organic noise floor.
    - Clone voices: zero/unnatural jitter, phase jumps every 20ms, near-zero noise floor.
    """
    num_samples = int(sr * duration)
    t = np.arange(num_samples) / sr
    
    # 1. Pitch Jitter (Fundamental Frequency f0 variation)
    if not is_clone:
        # Organic jitter: slow organic drifts (simulating muscle tremor) + small random noise
        f0_drift = 3.0 * np.sin(2 * np.pi * 1.5 * t) + 1.5 * np.cos(2 * np.pi * 3.7 * t)
        f0_jitter = np.random.normal(0, 0.4, num_samples)
        f0_t = base_f0 + f0_drift + f0_jitter
    else:
        # Clone: perfectly constant pitch (no jitter) or a rigid artificial modulation
        f0_t = base_f0 * np.ones(num_samples)
        
    # 2. Phase Synthesis
    if not is_clone:
        # Smooth phase: cumulative sum of the frequency
        phase = 2 * np.pi * np.cumsum(f0_t) / sr
    else:
        # Clone: frame-by-frame synthesis (20ms frames) causing phase jumps
        phase = np.zeros(num_samples)
        frame_size = int(0.020 * sr)  # 20ms = 320 samples
        current_phase = 0.0
        
        for i in range(0, num_samples, frame_size):
            # Introduce a random phase jump at the start of each frame
            phase_jump = np.random.uniform(0.1 * np.pi, 1.9 * np.pi)
            current_phase += phase_jump
            
            frame_len = min(frame_size, num_samples - i)
            t_frame = np.arange(frame_len) / sr
            # Accumulate phase within the frame smoothly, but with the jump offset
            phase[i:i+frame_len] = 2 * np.pi * f0_t[i:i+frame_len] * t_frame + current_phase
            
            # Update current phase for the next frame
            current_phase = phase[i+frame_len-1]

    # 3. Waveform Synthesis (Fundamental + Harmonics to sound speech-like)
    y = np.sin(phase) + 0.6 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    
    # 4. Amplitude envelope (vocal rise and decay to sound like speech)
    envelope = np.sin(np.pi * t / duration) ** 0.5
    y = y * envelope
    
    # 5. Noise Floor (Background noise)
    if not is_clone:
        # Real call: higher noise floor (room/channel noise)
        noise = np.random.normal(0, 0.03, num_samples)
    else:
        # Clone: digital silence / extremely low noise floor
        noise = np.random.normal(0, 0.0005, num_samples)
        
    y = y + noise
    
    # Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    return y

def save_wav(filepath, signal, sr=16000):
    """Saves a NumPy float32 array as a 16-bit PCM WAV file using Python standard library."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    signal_int = (signal * 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as w:
        w.setnchannels(1)        # Mono
        w.setsampwidth(2)        # 16-bit (2 bytes)
        w.setframerate(sr)       # 16kHz
        w.writeframes(signal_int.tobytes())

def build_simulated_dataset(base_dir="data"):
    """
    Creates real and cloned folders for:
    - vishal (base F0 = 120Hz, male)
    - abhinav (base F0 = 135Hz, male)
    - aditya (base F0 = 150Hz, male)
    - dhruv (base F0 = 110Hz, male)
    """
    speakers = {
        "vishal": 120.0,
        "abhinav": 135.0,
        "aditya": 150.0,
        "dhruv": 110.0
    }
    
    samples_per_speaker = 50
    print("Generating synthetic datasets...")
    
    for speaker, base_f0 in speakers.items():
        print(f"Generating voice data for speaker: {speaker} (f0={base_f0}Hz)")
        
        # 1. Real Voices
        real_dir = os.path.join(base_dir, "real_voices", speaker)
        for i in range(1, samples_per_speaker + 1):
            sig = generate_voice_signal(is_clone=False, base_f0=base_f0)
            filename = f"{speaker}_{i:03d}.wav"
            save_wav(os.path.join(real_dir, filename), sig)
            
        # 2. Clones
        clone_dir = os.path.join(base_dir, "clones", speaker)
        for i in range(1, samples_per_speaker + 1):
            sig = generate_voice_signal(is_clone=True, base_f0=base_f0)
            filename = f"{speaker}_clone_{i:03d}.wav"
            save_wav(os.path.join(clone_dir, filename), sig)
            
    print("\nGeneration Complete!")
    print(f"Real voices stored in: {os.path.join(base_dir, 'real_voices/')}")
    print(f"Clone voices stored in: {os.path.join(base_dir, 'clones/')}")

if __name__ == "__main__":
    build_simulated_dataset()
