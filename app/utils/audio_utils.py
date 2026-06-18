"""
app/utils/audio_utils.py

Audio I/O and preprocessing helpers shared across the ML service layer.

Responsibilities:
- Persist uploaded audio (UploadFile) to a temporary WAV file on disk.
- Load audio into a normalized, mono, 16kHz waveform tensor suitable for
  SpeechBrain's ECAPA-TDNN model.
- Validate file extensions and clean up temporary files.
"""

import os
import uuid
from pathlib import Path

import torch
import torchaudio
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.logging import get_logger
from app.utils.exceptions import InvalidAudioFileError

settings = get_settings()
log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def validate_audio_extension(filename: str | None) -> str:
    """
    Validate that an uploaded file has an allowed audio extension.

    Returns the lowercase extension (including the leading dot) if valid,
    otherwise raises InvalidAudioFileError.
    """
    if not filename:
        raise InvalidAudioFileError(filename or "<unknown>", "missing filename")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidAudioFileError(
            filename,
            f"unsupported extension '{ext}'. "
            f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


async def save_upload_to_temp(upload_file: UploadFile) -> str:
    """
    Persist an UploadFile to a uniquely named temporary file on disk.

    Returns the absolute path to the saved file. Caller is responsible for
    deleting the file via `cleanup_temp_file` once processing is complete.
    """
    ext = validate_audio_extension(upload_file.filename)

    temp_dir = Path(settings.TEMP_UPLOAD_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = temp_dir / temp_filename

    contents = await upload_file.read()

    if len(contents) == 0:
        raise InvalidAudioFileError(upload_file.filename or "<unknown>", "file is empty")

    if len(contents) > settings.max_upload_size_bytes:
        raise InvalidAudioFileError(
            upload_file.filename or "<unknown>",
            f"file exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    with open(temp_path, "wb") as f:
        f.write(contents)

    # Reset the file pointer in case the caller needs to read it again.
    await upload_file.seek(0)

    log.debug(f"Saved upload '{upload_file.filename}' -> '{temp_path}'")
    return str(temp_path)


def cleanup_temp_file(path: str | None) -> None:
    """Best-effort removal of a temporary file. Never raises."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            log.debug(f"Removed temp file '{path}'")
    except OSError as exc:
        log.warning(f"Failed to remove temp file '{path}': {exc}")


def load_waveform(
    file_path: str, target_sr: int | None = None
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file as a mono waveform tensor resampled to `target_sr`.

    Returns a tuple of (waveform, sample_rate) where waveform has shape
    (1, num_samples) — the format expected by SpeechBrain's
    `encode_batch`.

    Raises:
        InvalidAudioFileError: if the file cannot be loaded/decoded.
    """
    target_sr = target_sr or settings.TARGET_SAMPLE_RATE

    try:
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception:  # noqa: BLE001
            # torchaudio >= 2.9 needs the optional torchcodec package as its
            # default backend. Fall back to soundfile (always present via
            # libsndfile) for WAV/FLAC, which covers all PhaseGuard use cases.
            import soundfile as sf
            data, sr_sf = sf.read(file_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)
            sample_rate = sr_sf
    except Exception as exc:  # noqa: BLE001 - surface as domain error
        raise InvalidAudioFileError(
            os.path.basename(file_path), f"could not decode audio ({exc})"
        ) from exc

    if waveform.numel() == 0:
        raise InvalidAudioFileError(
            os.path.basename(file_path), "decoded audio contains zero samples"
        )

    # Convert multi-channel audio to mono by averaging channels.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if the source sample rate differs from the target.
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )
        waveform = resampler(waveform)
        sample_rate = target_sr

    # Peak-normalize so the loudest sample has amplitude 1.0. This makes
    # recordings captured at different volumes comparable, as recommended
    # in the PhaseGuard architecture spec.
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    return waveform, sample_rate
