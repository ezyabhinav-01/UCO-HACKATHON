"""
streamlit_app/app.py

PhaseGuard - Layer 2 (Speaker Verification) Streamlit Frontend.

This is the landing page. Additional pages (Enrollment, Verification,
History, Risk Dashboard) live in streamlit_app/pages/ and are picked up
automatically by Streamlit's multipage app support.

Run with:
    streamlit run streamlit_app/app.py
"""

import streamlit as st

import api_client
from config import (
    API_BASE_URL,
    LAYER1_FRAUD_THRESHOLD,
    MIN_ENROLLMENT_RECORDINGS,
    SIMILARITY_THRESHOLD,
)

st.set_page_config(
    page_title="PhaseGuard - Layer 2",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ PhaseGuard — Layer 2: Speaker Verification")
st.caption("Identity verification powered by SpeechBrain ECAPA-TDNN")

st.markdown(
    """
PhaseGuard's **Layer 2** verifies that a live caller is who they claim to
be, using a 192-dimensional speaker embedding (voiceprint) extracted by a
pretrained **ECAPA-TDNN** model and compared via **cosine similarity**.

Use the pages in the sidebar to:

- **Enrollment** — upload 30-50 voice recordings for a user and build their
  voiceprint.
- **Verification** — upload a live audio sample and check it against an
  enrolled voiceprint.
- **History** — review past verification attempts and risk evaluations for
  a user.
- **Risk Dashboard** — visualize risk scores and decisions over time.
"""
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Similarity Threshold", f"{SIMILARITY_THRESHOLD:.2f}")
col2.metric("Layer 1 Fraud Threshold", f"{LAYER1_FRAUD_THRESHOLD:.2f}")
col3.metric("Min. Enrollment Recordings", MIN_ENROLLMENT_RECORDINGS)
col4.metric("API Base URL", API_BASE_URL)

st.divider()

st.subheader("Backend connectivity")

if st.button("Check API health"):
    ok, payload = api_client.check_health()
    if ok:
        st.success("API is reachable.")
        st.json(payload)
    else:
        st.error(f"API health check failed: {payload}")

st.divider()

with st.expander("How the pipeline works", expanded=False):
    st.markdown(
        """
**Enrollment flow**

1. Record 30-50 utterances per user (WAV, 16kHz recommended).
2. Each recording is converted to a 192-dim ECAPA-TDNN embedding.
3. Embeddings are averaged into a single voiceprint.
4. The voiceprint is stored in PostgreSQL via pgvector.

**Verification flow**

1. Live caller audio is captured.
2. An ECAPA-TDNN embedding is generated for the live audio.
3. The stored voiceprint is fetched for the claimed user.
4. Cosine similarity is computed between the two embeddings.
5. The similarity score and Layer 1 (AI-voice-detection) score are passed
   to the **Risk Engine**:
   - `layer1_score > 0.70` → **FRAUD_ALERT**
   - `speaker_similarity < 0.65` → **FRAUD_ALERT**
   - otherwise → **CLEAN**
"""
    )
