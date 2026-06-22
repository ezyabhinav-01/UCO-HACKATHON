"""
tests/test_audio_utils.py

Unit tests for app.utils.audio_utils.

Tests file-extension validation and the load_waveform / save-to-temp
helpers without requiring a live database or ML model. Where I/O is
needed, small synthetic WAV files are generated in-memory.
"""

import io
import os
import struct
import tempfile
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from app.utils.audio_utils import (
    cleanup_temp_file,
    load_waveform,
    validate_audio_extension,
)
from app.utils.exceptions import InvalidAudioFileError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    """Return a minimal valid WAV file as bytes (single-channel, 16-bit PCM)."""
    num_samples = int(sr * duration_sec)
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples))
               * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _write_temp_wav(duration_sec: float = 1.0, sr: int = 16000) -> str:
    """Write a synthetic WAV to a temp file and return the path."""
    wav_bytes = _make_wav_bytes(duration_sec, sr)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path


# ---------------------------------------------------------------------------
# validate_audio_extension
# ---------------------------------------------------------------------------

class TestValidateAudioExtension:
    def test_wav_is_accepted(self):
        assert validate_audio_extension("voice.wav") == ".wav"

    def test_flac_is_accepted(self):
        assert validate_audio_extension("voice.flac") == ".flac"

    def test_mp3_is_accepted(self):
        assert validate_audio_extension("voice.mp3") == ".mp3"

    def test_ogg_is_accepted(self):
        assert validate_audio_extension("voice.ogg") == ".ogg"

    def test_case_insensitive(self):
        assert validate_audio_extension("voice.WAV") == ".wav"
        assert validate_audio_extension("voice.Mp3") == ".mp3"

    def test_pdf_is_rejected(self):
        with pytest.raises(InvalidAudioFileError):
            validate_audio_extension("document.pdf")

    def test_exe_is_rejected(self):
        with pytest.raises(InvalidAudioFileError):
            validate_audio_extension("malware.exe")

    def test_none_filename_is_rejected(self):
        with pytest.raises(InvalidAudioFileError):
            validate_audio_extension(None)

    def test_empty_string_filename_is_rejected(self):
        with pytest.raises(InvalidAudioFileError):
            validate_audio_extension("")


# ---------------------------------------------------------------------------
# load_waveform
# ---------------------------------------------------------------------------

class TestLoadWaveform:
    def test_loads_valid_wav(self):
        path = _write_temp_wav(duration_sec=2.0, sr=16000)
        try:
            waveform, sr = load_waveform(path)
            assert isinstance(waveform, torch.Tensor)
            assert waveform.shape[0] == 1          # mono
            assert waveform.shape[1] > 0
            assert sr == 16000
        finally:
            cleanup_temp_file(path)

    def test_resamples_to_target_sr(self):
        # Write at 44100 Hz, load expecting 16000 Hz
        path = _write_temp_wav(duration_sec=1.0, sr=44100)
        try:
            waveform, sr = load_waveform(path, target_sr=16000)
            assert sr == 16000
            # At 1 second, 16000-sample target → approximately 16000 samples
            assert 15000 < waveform.shape[1] < 17000
        finally:
            cleanup_temp_file(path)

    def test_peak_normalizes_output(self):
        path = _write_temp_wav(duration_sec=1.0, sr=16000)
        try:
            waveform, _ = load_waveform(path)
            max_val = waveform.abs().max().item()
            # After normalization the peak should be ≈ 1.0
            assert max_val == pytest.approx(1.0, abs=1e-4)
        finally:
            cleanup_temp_file(path)

    def test_invalid_path_raises(self):
        with pytest.raises(InvalidAudioFileError):
            load_waveform("/nonexistent/path/audio.wav")

    def test_empty_file_raises(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with pytest.raises(InvalidAudioFileError):
                load_waveform(path)
        finally:
            cleanup_temp_file(path)


# ---------------------------------------------------------------------------
# cleanup_temp_file
# ---------------------------------------------------------------------------

class TestCleanupTempFile:
    def test_removes_existing_file(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        assert os.path.exists(path)
        cleanup_temp_file(path)
        assert not os.path.exists(path)

    def test_does_not_raise_for_nonexistent_path(self):
        # Should silently succeed without raising
        cleanup_temp_file("/nonexistent/path/file.wav")

    def test_does_not_raise_for_none(self):
        cleanup_temp_file(None)
