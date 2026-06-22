"""
streamlit_app/config.py

Shared configuration for the Streamlit frontend.

The API base URL is read from the API_BASE_URL environment variable so the
same code works both locally (http://localhost:8000) and inside Docker
Compose (http://api:8000).
"""

import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE_URL}/api/v1"

SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.65"))
LAYER1_FRAUD_THRESHOLD = float(os.environ.get("LAYER1_FRAUD_THRESHOLD", "0.70"))
MIN_ENROLLMENT_RECORDINGS = int(os.environ.get("MIN_ENROLLMENT_RECORDINGS", "3"))
