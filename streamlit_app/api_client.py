"""
streamlit_app/api_client.py

Thin HTTP client wrapping calls to the PhaseGuard Layer 2 FastAPI backend.

Every function returns a tuple of (success: bool, data_or_error). On
success, `data_or_error` is the parsed JSON response. On failure, it is a
human-readable error message extracted from the API's error response (or a
generic message if the API was unreachable).

Keeping all HTTP calls in one module makes it trivial to swap the backend
URL, add auth headers, or add retry logic later without touching every page.
"""

from typing import Any

import requests

from config import API_V1

DEFAULT_TIMEOUT = 30  # seconds; embedding extraction can take a few seconds


def _handle_response(response: requests.Response) -> tuple[bool, Any]:
    """Convert a requests.Response into (success, payload_or_error_message)."""
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text or "Empty response from server"}

    if response.ok:
        return True, payload

    detail = payload.get("detail", f"Request failed with status {response.status_code}")
    if isinstance(detail, list):
        # FastAPI validation errors are returned as a list of dicts.
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return False, str(detail)


def create_user(name: str, email: str) -> tuple[bool, Any]:
    """POST /api/v1/users"""
    try:
        response = requests.post(
            f"{API_V1}/users",
            json={"name": name, "email": email},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_user(user_id: str) -> tuple[bool, Any]:
    """GET /api/v1/users/{id}"""
    try:
        response = requests.get(f"{API_V1}/users/{user_id}", timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def enroll_user(user_id: str, files: list[tuple[str, bytes, str]]) -> tuple[bool, Any]:
    """
    POST /api/v1/enroll

    Args:
        user_id: UUID of the user to enroll.
        files: list of (filename, file_bytes, mime_type) tuples for each
               recording.
    """
    try:
        multipart_files = [
            ("files", (filename, content, mime_type))
            for filename, content, mime_type in files
        ]
        response = requests.post(
            f"{API_V1}/enroll",
            data={"user_id": user_id},
            files=multipart_files,
            timeout=120,  # enrollment processes many files; allow extra time
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def verify_user(
    user_id: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
    layer1_score: float = 0.0,
) -> tuple[bool, Any]:
    """POST /api/v1/verify"""
    try:
        response = requests.post(
            f"{API_V1}/verify",
            data={"user_id": user_id, "layer1_score": str(layer1_score)},
            files={"file": (filename, file_bytes, mime_type)},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_verification_history(user_id: str, limit: int = 50) -> tuple[bool, Any]:
    """GET /api/v1/verification-history/{id}"""
    try:
        response = requests.get(
            f"{API_V1}/verification-history/{user_id}",
            params={"limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_risk_history(user_id: str, limit: int = 50) -> tuple[bool, Any]:
    """GET /api/v1/risk-history/{id}"""
    try:
        response = requests.get(
            f"{API_V1}/risk-history/{user_id}",
            params={"limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def check_health() -> tuple[bool, Any]:
    """GET /health"""
    try:
        response = requests.get(
            f"{API_V1.rsplit('/api/v1', 1)[0]}/health", timeout=10
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)
