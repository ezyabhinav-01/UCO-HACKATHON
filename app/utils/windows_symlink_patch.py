"""
app/utils/windows_symlink_patch.py

Windows symlink-permission workaround.

SpeechBrain's model-fetching code (speechbrain.utils.fetching.fetch) creates
a symlink from its Hugging Face cache to the configured save directory after
downloading each model file. On Windows, creating a symlink requires either:
  - Administrator privileges, or
  - Developer Mode enabled (Settings > Privacy & Security > For developers)

Without one of those, Path.symlink_to() raises:
    OSError: [WinError 1314] A required privilege is not held by the client

This is a hard requirement enforced by Windows itself, not something
SpeechBrain or huggingface_hub expose a config flag for (verified against
SpeechBrain 1.0.0's fetching.py source, which calls symlink_to()
unconditionally with no environment-variable or parameter override).

Rather than requiring every teammate to enable Developer Mode or always run
as Administrator, this module monkeypatches `pathlib.Path.symlink_to` so
that if symlink creation fails specifically due to a permissions error
(WinError 1314 / errno EPERM), it transparently falls back to copying the
file instead. This has no effect on Linux/Mac (where symlinks work
normally for unprivileged users) and is a no-op there.
"""

import errno
import os
import pathlib
import shutil

from app.core.logging import get_logger

log = get_logger(__name__)

_original_symlink_to = pathlib.Path.symlink_to
_patch_applied = False


def _symlink_to_with_copy_fallback(self, target, target_is_directory=False):
    """
    Drop-in replacement for Path.symlink_to that falls back to a file copy
    if symlink creation fails due to insufficient privileges.
    """
    try:
        _original_symlink_to(self, target, target_is_directory=target_is_directory)
    except OSError as exc:
        # WinError 1314 = "A required privilege is not held by the client".
        # Also guard on errno.EPERM in case a different platform/Python
        # version surfaces this as a generic permission error.
        is_windows_privilege_error = (
            getattr(exc, "winerror", None) == 1314 or exc.errno == errno.EPERM
        )
        if not is_windows_privilege_error:
            raise

        log.warning(
            f"Symlink creation failed due to insufficient privileges "
            f"('{self}' -> '{target}'). Falling back to copying the file "
            "instead. To avoid this fallback (and save a small amount of "
            "disk space), enable Windows Developer Mode: Settings > "
            "Privacy & Security > For developers > Developer Mode."
        )

        target_path = pathlib.Path(target)
        if target_path.is_dir():
            shutil.copytree(target_path, self)
        else:
            # Ensure parent directory exists before copying.
            self.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, self)


def apply_windows_symlink_fallback() -> None:
    """
    Monkeypatch pathlib.Path.symlink_to with the copy-fallback version.

    Idempotent — safe to call multiple times (e.g. if main.py and a test
    both import this module). Only has any practical effect on Windows;
    on Linux/Mac, unprivileged symlink creation works normally so the
    fallback branch is never triggered.
    """
    global _patch_applied
    if _patch_applied:
        return

    if os.name == "nt":
        pathlib.Path.symlink_to = _symlink_to_with_copy_fallback
        log.debug("Applied Windows symlink-to-copy fallback patch.")

    _patch_applied = True
