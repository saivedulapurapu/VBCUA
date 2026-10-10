import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import api
from api import BackendError
from display import (
    FOUND, MISSING, PLAIN, concept_title as get_concept_title, description_html, details_rows,
    demote_headings, feedback_in_progress, feedback_notice, grade_badge_color, history_rows, initial_concept_id, keyword_badges,
    keyword_badges_html, score_cards, transcript_html,
)

st.set_page_config(
    page_title="VBCUA · Concept Understanding Analyser",
    page_icon=":material/record_voice_over:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------- Helpers ----------

def show_error(error: BackendError):
    """User-facing message, with the technical detail tucked into an expander."""
    st.error(error.message)
    if error.detail:
        with st.expander("Technical details"):
            st.code(error.detail, language=None)


def show_evaluation(result):
    """Display an attempt and remember it in the URL so a page reload keeps it."""
    st.session_state.current_evaluation = result
    st.query_params["attempt"] = str(result["id"])


@st.cache_data(ttl=600, show_spinner=False)
def cached_pdf(eval_id: int, feedback_source: str) -> bytes:
    """Fetched once per attempt and feedback version (the report changes when AI feedback arrives)."""
    return api.get_pdf(eval_id)


def wait_for_feedback(eval_id: int, max_wait_s: int = 300):
    """
    Requests AI feedback, or waits for a request already running elsewhere
    (the backend never calls Gemini twice for the same attempt).
    """
    result = api.request_feedback(eval_id)
    start = time.time()
    while result.get("feedback_status") == "generating" and time.time() - start < max_wait_s:
        time.sleep(3)
        result = api.get_evaluation(eval_id)
    return result


def step_title(number: int, text: str):
    st.html(f"<p class='step-title'><span class='step-number'>{number}</span>{text}</p>")


def run_evaluation_with_progress(concept, audio_bytes, filename):
    """Runs the (slow) evaluation in a worker thread while showing elapsed time."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(api.evaluate, concept["id"], audio_bytes, filename)
        with st.status(f"Analysing your explanation of {concept['title']}…", expanded=True) as status:
            st.write("Transcribing speech, scoring understanding and fluency, and writing feedback. "
                     "This usually takes 1–2 minutes for a one-minute recording.")
            elapsed = st.empty()
            start = time.time()
            while not future.done():
                elapsed.caption(f"Elapsed: {int(time.time() - start)} s")
                time.sleep(1)
            try:
                result = future.result()
            except BackendError as e:
                status.update(label="Analysis failed", state="error", expanded=True)
                show_error(e)
                return None
            status.update(label=f"Analysis complete ({int(time.time() - start)} s)", state="complete", expanded=False)
            return result


def waveform_figure(audio_bytes, filename):
    import librosa
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        y, sr = librosa.load(path, sr=None)
    finally:
        os.remove(path)
    fig, ax = plt.subplots(figsize=(6, 1.6))
    t = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(t, y, color="#2563eb", linewidth=0.6)
    ax.set_xlabel("Time (s)", fontsize=8, color="#475569")
    ax.set_yticks([])
    ax.tick_params(labelsize=8, colors="#475569")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


# ---------- Data ----------

try:
    concepts = api.get_concepts()
    backend_error = None
except BackendError as e:
    concepts, backend_error = [], e

# Restore the attempt shown before a page reload (?attempt=<id>)
retry_id = st.session_state.pop("feedback_retry", None)
if retry_id and st.session_state.get("current_evaluation", {}).get("id") == retry_id:
    st.session_state.current_evaluation = {**st.session_state.current_evaluation, "feedback_status": "pending"}

if "current_evaluation" not in st.session_state and st.query_params.get("attempt"):
    try:
        st.session_state.current_evaluation = api.get_evaluation(int(st.query_params["attempt"]))
    except (BackendError, ValueError):
        st.query_params.pop("attempt", None)


# ---------- Sidebar: add a concept ----------

with st.sidebar:
    st.subheader("Add a concept")
    st.caption("Add your own topic with a reference definition and the keywords a good explanation should mention.")
    with st.form("new_concept", clear_on_submit=True):
        new_title = st.text_input("Title", placeholder="e.g. REST API")
        new_desc = st.text_area("Reference definition")
        new_kws = st.text_input("Keywords (comma separated)", placeholder="e.g. client, server, HTTP")
        if st.form_submit_button("Add concept", type="primary"):
            keywords = [kw.strip() for kw in new_kws.split(",") if kw.strip()]
            if new_title.strip() and new_desc.strip() and keywords:
                try:
                    created = api.create_concept(new_title.strip(), new_desc.strip(), keywords)
                    st.session_state.concept_id = created["id"]
                    st.rerun()
                except BackendError as e:
                    show_error(e)
            else:
                st.warning("Please fill in the title, definition and at least one keyword.")


# ---------- Header ----------

st.html(
    "<div class='app-eyebrow'>VBCUA</div>"
    "<h1 class='app-title'>Concept Understanding Analyser</h1>"
    "<p class='app-subtitle'>Explain a computing concept out loud and get feedback on your understanding and delivery.</p>"
)

if backend_error:
    show_error(backend_error)

col_input, col_results = st.columns([5, 7], gap="large")
feedback_slot = None


# ---------- Steps 1 and 2: input ----------

with col_input:
    with st.container(border=True):
        step_title(1, "Choose a concept")
        if concepts:
            by_id = {c["id"]: c for c in concepts}
            # Keep the choice across reruns and page reloads (?concept=<id>)
            if st.session_state.get("concept_id") not in by_id:
                st.session_state.concept_id = initial_concept_id(concepts, st.query_params.get("concept"))
            st.selectbox(
                "Concept", options=list(by_id), format_func=lambda cid: by_id[cid]["title"],
                key="concept_id", label_visibility="collapsed",
            )
            st.query_params["concept"] = str(st.session_state.concept_id)
            selected_concept = by_id[st.session_state.concept_id]
            with st.expander("Reference definition and keywords"):
                st.html(description_html(selected_concept["description"]))
                st.html(keyword_badges_html([(kw, PLAIN) for kw in selected_concept["keywords"]]))
        else:
            selected_concept = None
            st.info("No concepts available yet. Add one from the sidebar.")

        st.divider()
        step_title(2, "Record or upload your explanation")
        mode = st.segmented_control(
            "Input", ["Record", "Upload"], default="Record", key="input_mode", label_visibility="collapsed",
        )
        audio_data, filename = None, "recording.wav"
        if mode == "Upload":
            uploaded = st.file_uploader("Audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
            if uploaded is not None:
                audio_data, filename = uploaded.read(), uploaded.name
        else:
            recording = st.audio_input("Click the microphone to start, and again to stop")
            if recording is not None:
                audio_data = recording.read()
        st.caption("Aim for 60–120 seconds, in a quiet room.")

        if audio_data:
            with st.expander("Waveform"):
                try:
                    st.pyplot(waveform_figure(audio_data, filename), clear_figure=True)
                except Exception:
                    st.caption("Waveform preview is not available for this file.")

        can_analyse = bool(audio_data and selected_concept)
        label = f"Analyse explanation of {selected_concept['title']}" if selected_concept else "Analyse explanation"
        # The button names the concept so the explanation is never graded against the wrong one
        if st.button(label, type="primary", width="stretch", disabled=not can_analyse):
            result = run_evaluation_with_progress(selected_concept, audio_data, filename)
            if result:
                show_evaluation(result)
                st.rerun()


# ---------- Step 3: results ----------

with col_results:
    with st.container(border=True):
        if "current_evaluation" in st.session_state:
            eval_res = st.session_state.current_evaluation
            concept_title = get_concept_title(eval_res)
            grade = eval_res["understanding_grade"]

            step_title(3, "Results")
            head_left, head_right = st.columns([3, 2], vertical_alignment="center")
            head_left.markdown(f"**{concept_title}**")
            head_left.caption(f"Attempt #{eval_res['id']} · {str(eval_res.get('created_at', ''))[:16].replace('T', ' ')}")
            with head_right:
                st.badge(grade, color=grade_badge_color(grade))

            for column, card in zip(st.columns(3), score_cards(eval_res)):
                with column:
                    st.metric(card["label"], card["value"], help=card["help"], border=True)
                    st.caption(card["caption"])

            if eval_res.get("recording_warning"):
                st.warning(eval_res["recording_warning"], icon=":material/mic_off:")
            notice = feedback_notice(eval_res)
            if notice:
                st.info(notice, icon=":material/info:")

            tab_feedback, tab_keywords, tab_transcript, tab_details = st.tabs(
                ["Feedback", "Keywords", "Transcript", "Details"]
            )
            with tab_feedback:
                if feedback_in_progress(eval_res):
                    # Filled at the end of the script, after the rest of the page has rendered
                    feedback_slot = st.empty()
                    feedback_slot.info("Writing feedback…", icon=":material/edit_note:")
                else:
                    st.markdown(demote_headings(eval_res["feedback"]))
                    if eval_res.get("feedback_status") == "failed" and st.button("Try AI feedback again"):
                        st.session_state.feedback_retry = eval_res["id"]
                        st.rerun()
            with tab_keywords:
                badges = keyword_badges(eval_res)
                found = sum(1 for _, status in badges if status == FOUND)
                if badges and all(status not in (FOUND, MISSING) for _, status in badges):
                    st.caption("Keyword matches were not stored for this older attempt.")
                elif badges:
                    st.caption(f"{found} of {len(badges)} key terms mentioned (whole-word matches).")
                if badges:
                    st.html(keyword_badges_html(badges))
            with tab_transcript:
                st.html(transcript_html(eval_res["transcription"]))
                st.caption(f"Transcribed by {eval_res.get('asr_model') or 'unknown model'}. "
                           "Filler words such as 'um' are usually not transcribed.")
            with tab_details:
                st.dataframe(details_rows(eval_res), hide_index=True, width="stretch")

            try:
                st.download_button(
                    "Download PDF report", data=cached_pdf(eval_res["id"], str(eval_res.get("feedback_source"))),
                    file_name=f"VBCUA_Report_{concept_title.replace(' ', '_')}_{eval_res['id']}.pdf",
                    mime="application/pdf", icon=":material/download:",
                )
            except BackendError as e:
                show_error(e)
        else:
            st.html(
                "<div class='empty-state'><h3>No results yet</h3>"
                "<p>Choose a concept, record or upload your explanation, then select Analyse.</p></div>"
            )


# ---------- History ----------

st.subheader("History")
try:
    history = api.get_evaluations()
except BackendError as e:
    history = []
    show_error(e)

if history:
    rows = history_rows(history)
    by_eval_id = {item["id"]: item for item in history}
    st.caption("Select a row to open that attempt.")
    event = st.dataframe(
        [{k: v for k, v in row.items() if k != "id"} for row in rows],
        hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="history_table",
        column_config={
            "Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%.1f"),
            "Duration (s)": st.column_config.NumberColumn("Duration (s)", format="%.0f"),
        },
    )
    selected = event.selection.rows
    selected_id = rows[selected[0]]["id"] if selected else None
    # Load only when the selection changes, so a new analysis is not replaced by an old row
    if selected_id is not None and selected_id != st.session_state.get("history_selected_id"):
        st.session_state.history_selected_id = selected_id
        show_evaluation(by_eval_id[selected_id])
        st.rerun()
    st.session_state.history_selected_id = selected_id
else:
    st.caption("No attempts yet. Your past attempts will appear here.")


# ---------- AI feedback (after the page has rendered, so scores are visible meanwhile) ----------

if feedback_slot is not None:
    eval_id = st.session_state.current_evaluation["id"]
    with feedback_slot.container():
        try:
            with st.spinner("Writing feedback…"):
                updated = wait_for_feedback(eval_id)
        except BackendError as e:
            updated = None
            show_error(e)
    if updated:
        show_evaluation(updated)
        st.rerun()
