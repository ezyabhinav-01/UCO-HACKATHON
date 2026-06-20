"""
streamlit_app/pages/1_Enrollment.py

Enrollment page: create a user (if needed) and upload 30-50 voice
recordings to build their ECAPA-TDNN voiceprint via POST /api/v1/enroll.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import api_client
from config import MIN_ENROLLMENT_RECORDINGS

st.set_page_config(page_title="PhaseGuard - Enrollment", page_icon="🎙️", layout="wide")

st.title("🎙️ Enrollment")
st.caption("Build a customer's voiceprint from multiple recordings")

# ----------------------------------------------------------------------
# Step 1: Create or select a user
# ----------------------------------------------------------------------
st.subheader("1. Create a user")

with st.form("create_user_form"):
    col1, col2 = st.columns(2)
    name = col1.text_input("Full name", placeholder="e.g. Vishal Kumar")
    email = col2.text_input("Email", placeholder="e.g. vishal@example.com")
    submitted = st.form_submit_button("Create user")

if submitted:
    if not name or not email:
        st.warning("Please provide both a name and an email.")
    else:
        ok, payload = api_client.create_user(name, email)
        if ok:
            st.success(f"User created: {payload['id']}")
            st.session_state["enroll_user_id"] = payload["id"]
            st.json(payload)
        else:
            st.error(f"Failed to create user: {payload}")

st.divider()

# ----------------------------------------------------------------------
# Step 2: Look up an existing user by id
# ----------------------------------------------------------------------
st.subheader("2. Select user to enroll")

default_user_id = st.session_state.get("enroll_user_id", "")
user_id = st.text_input("User ID (UUID)", value=default_user_id)

if user_id:
    ok, payload = api_client.get_user(user_id)
    if ok:
        st.session_state["enroll_user_id"] = user_id
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Name", payload["name"])
        col2.metric("Email", payload["email"])
        col3.metric("Enrolled?", "Yes" if payload["is_enrolled"] else "No")
        col4.metric("Recordings on file", payload.get("recording_count", 0))
    else:
        st.error(f"Could not find user: {payload}")

st.divider()

# ----------------------------------------------------------------------
# Step 3: Upload recordings and enroll
# ----------------------------------------------------------------------
st.subheader("3. Upload voice recordings")

st.info(
    f"Upload at least **{MIN_ENROLLMENT_RECORDINGS}** recordings "
    "(30-50 recommended for a stable voiceprint). WAV format, 16kHz mono "
    "preferred."
)

uploaded_files = st.file_uploader(
    "Upload WAV/FLAC/MP3 recordings",
    type=["wav", "flac", "mp3", "ogg", "m4a"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)}** file(s) selected.")

enroll_clicked = st.button(
    "🚀 Run Enrollment", type="primary", disabled=not (user_id and uploaded_files)
)

if enroll_clicked:
    files_payload = [
        (f.name, f.getvalue(), f.type or "audio/wav") for f in uploaded_files
    ]

    with st.spinner(
        f"Extracting ECAPA-TDNN embeddings for {len(files_payload)} recordings "
        "and building voiceprint..."
    ):
        ok, payload = api_client.enroll_user(user_id, files_payload)

    if ok:
        st.success(payload["message"])
        col1, col2 = st.columns(2)
        col1.metric("Recordings processed", payload["recording_count"])
        col2.metric("Embedding dimension", payload["embedding_dimension"])
        st.json(payload)
    else:
        st.error(f"Enrollment failed: {payload}")
