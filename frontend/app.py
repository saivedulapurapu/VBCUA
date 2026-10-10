import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
import api
from api import BackendError
from display import (
    FOUND, MISSING, PLAIN, concept_title as get_concept_title, description_html, esc,
    feedback_notice, fluency_text, history_rows, initial_concept_id, keyword_badges,
    keyword_badges_html, transcript_html,
)

# Load local environment variables (for Gemini key check)
load_dotenv()

st.set_page_config(
    page_title="VBCUA - Voice-Based Concept Understanding Analyser",
    layout="wide",
    page_icon="🎙️",
    initial_sidebar_state="expanded"
)

# Custom styles injection
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Custom CSS styling file not found. Rendering using default Streamlit theme.")

# Title Banner
st.markdown("<div class='neon-title'>VBCUA Assessment Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-subtitle'>AI-Powered Semantic & Speech Fluency Concept Evaluation</div>", unsafe_allow_html=True)

# Helper Functions
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

try:
    concepts = api.get_concepts()
    backend_error = None
except BackendError as e:
    concepts, backend_error = [], e

# Restore the attempt shown before a page reload (?attempt=<id>)
if "current_evaluation" not in st.session_state and st.query_params.get("attempt"):
    try:
        st.session_state.current_evaluation = api.get_evaluation(int(st.query_params["attempt"]))
    except (BackendError, ValueError):
        st.query_params.pop("attempt", None)

# Sidebar configuration
with st.sidebar:
    st.markdown("### 📚 Practice Concept")
    
    if concepts:
        by_id = {c["id"]: c for c in concepts}
        # Keep the choice across reruns and page reloads (?concept=<id>)
        if st.session_state.get("concept_id") not in by_id:
            st.session_state.concept_id = initial_concept_id(concepts, st.query_params.get("concept"))
        st.selectbox(
            "Choose a concept to explain:", options=list(by_id),
            format_func=lambda cid: by_id[cid]["title"], key="concept_id",
        )
        st.query_params["concept"] = str(st.session_state.concept_id)
        selected_concept = by_id[st.session_state.concept_id]
        
        # Show Concept Details
        st.markdown("**📖 Description**")
        st.html(description_html(selected_concept["description"]))

        # Show Keywords
        st.markdown("**🔑 Required Keywords**")
        st.html(keyword_badges_html([(kw, PLAIN) for kw in selected_concept["keywords"]]))
    else:
        if backend_error:
            show_error(backend_error)
        else:
            st.info("No concepts yet. Add one below.")
        selected_concept = None

    st.markdown("---")
    
    # Hide System Configuration inside an expander so sidebar is clean
    with st.expander("🛠️ System Configuration", expanded=False):
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        new_gemini_key = st.text_input("Google Gemini API Key", value=gemini_key, type="password", help="Enables advanced qualitative AI analysis & summaries")
        
        if new_gemini_key != gemini_key:
            # Save key dynamically to .env
            try:
                with open(".env", "a") as f:
                    f.write(f"\nGEMINI_API_KEY={new_gemini_key}\n")
                st.success("API Key saved to .env! (Restart backend to apply)")
            except Exception as e:
                st.error(f"Failed to save key: {e}")

    st.markdown("---")
    st.markdown("### ➕ Add New Concept")
    with st.expander("Create Custom Topic"):
        new_title = st.text_input("Title (e.g. 'REST API')")
        new_desc = st.text_area("Ideal Definition")
        new_kws = st.text_input("Keywords (comma separated)")
        if st.button("Submit Concept"):
            if new_title and new_desc and new_kws:
                try:
                    created = api.create_concept(new_title, new_desc, [kw.strip() for kw in new_kws.split(",") if kw.strip()])
                    st.session_state.concept_id = created["id"]
                    st.rerun()
                except BackendError as e:
                    show_error(e)
            else:
                st.warning("Please fill all fields.")

# Main Dashboard Layout
col_left, col_right = st.columns([1, 1.2])

with col_left:
    with st.container(border=True):
        st.markdown("### 🎙️ Spoken Explanation Input")
        st.markdown("Explain the selected concept out loud. You can record directly using your microphone or upload an audio file.")
        
        audio_data = None
        filename = "recording.wav"
        
        # Audio recorder
        if hasattr(st, "audio_input"):
            st.markdown("**Option 1: Record live speech**")
            recording = st.audio_input("Click to start recording your voice")
            if recording is not None:
                audio_data = recording.read()
                filename = "recording.wav"
                st.audio(audio_data, format="audio/wav")
        else:
            st.info("Direct recording requires Streamlit 1.34+. Please use the file uploader below.")
        
        st.markdown("<br>**Option 2: Upload audio file (.wav, .mp3, .m4a)**", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload audio explanation", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            filename = uploaded_file.name
            st.audio(audio_data, format=f"audio/{os.path.splitext(filename)[1][1:]}")
            
        st.markdown("---")
        
        if audio_data and selected_concept:
            # The button names the concept so the explanation is never graded against the wrong one
            if st.button(f"🚀 Analyze explanation of {selected_concept['title']}"):
                result = run_evaluation_with_progress(selected_concept, audio_data, filename)
                if result:
                    show_evaluation(result)
                    st.rerun()
        elif not selected_concept:
            st.warning("Please create or select a concept in the sidebar before evaluating.")
        else:
            st.info("Record your explanation or upload a file to begin the evaluation.")

    # Waveform Display
    if audio_data:
        with st.container(border=True):
            st.markdown("### 📈 Spoken Audio Waveform")
            try:
                import librosa
                import io
                # We can save bytes to a temp file, load using librosa, and plot
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_audio:
                    temp_audio.write(audio_data)
                    temp_audio_path = temp_audio.name
                    
                y, sr = librosa.load(temp_audio_path, sr=None)
                os.remove(temp_audio_path)
                
                # Matplotlib dark waveform plot
                fig, ax = plt.subplots(figsize=(6, 2.2), facecolor='none')
                ax.set_facecolor('none')
                time = np.linspace(0, len(y)/sr, num=len(y))
                ax.plot(time, y, color='#60a5fa', linewidth=0.8, alpha=0.8)
                ax.fill_between(time, y, color='#4f46e5', alpha=0.15)
                
                # Styling
                ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.1))
                ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.1))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors=(1.0, 1.0, 1.0, 0.5), labelsize=8)
                ax.set_xlabel("Time (s)", color=(1.0, 1.0, 1.0, 0.5), fontsize=8)
                ax.set_ylabel("Amplitude", color=(1.0, 1.0, 1.0, 0.5), fontsize=8)
                ax.grid(True, color=(1.0, 1.0, 1.0, 0.05), linestyle='--')
                
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.warning(f"Waveform rendering skipped: {e}")

with col_right:
    if "current_evaluation" in st.session_state:
        eval_res = st.session_state.current_evaluation
        concept_title = get_concept_title(eval_res)
        
        with st.container(border=True):
            st.markdown(f"### 📊 Analysis Results: **{concept_title}**")
            
            # Dashboard KPIs
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.html(f"""
                <div class='metric-box purple'>
                    <div class='metric-label'>Overall Score</div>
                    <div class='metric-value purple'>{eval_res['final_score']:.1f}%</div>
                    <div style='font-size:0.8rem; color:#bd00ff;'>{esc(eval_res['understanding_grade'])}</div>
                </div>
                """)
                
            with kpi_col2:
                st.html(f"""
                <div class='metric-box blue'>
                    <div class='metric-label'>Semantic Match</div>
                    <div class='metric-value blue'>{eval_res['semantic_score']:.1f}%</div>
                    <div style='font-size:0.8rem; color:#00f0ff;'>Sentence-BERT Match</div>
                </div>
                """)
                
            with kpi_col3:
                # Stored backend fluency score (never recomputed here)
                fluency_display = fluency_text(eval_res)
                pause_display = eval_res.get("pause_ratio_v1")
                if pause_display is None:
                    pause_display = eval_res["pause_ratio"]
                st.html(f"""
                <div class='glass-card' style='padding:15px !important; margin:0 !important; border-color: rgba(0, 255, 102, 0.2) !important; text-align:center;'>
                    <div class='metric-label'>Speech Fluency</div>
                    <div class='metric-value green'>{esc(fluency_display)}</div>
                    <div style='font-size:0.8rem; color:#00ff66;'>P: {pause_display*100:.0f}% | Fillers (exp.): {esc(eval_res['filler_words_count'])}</div>
                </div>
                """)
            
            st.caption("Filler counts are experimental — speech-to-text usually removes 'um/uh'. Not used in scoring.")
            
            # Microphone check (from raw RMS; not part of any score)
            if eval_res.get("recording_warning"):
                st.warning(eval_res["recording_warning"], icon="🎙️")
                
            st.markdown("---")
            
            # Plotly Score Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = eval_res['final_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Comprehension Accuracy & Speech Mastery", 'font': {'size': 14, 'color': '#ffffff'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                    'bar': {'color': "#4f46e5"},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                        {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "#60a5fa", 'width': 4},
                        'thickness': 0.75,
                        'value': eval_res['final_score']
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#ffffff", 'family': "Outfit"},
                height=250,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
            # Keyword Coverage Visualization
            st.markdown("#### 🔑 Core Keyword Coverage Check")
            # Stored matches for the evaluated attempt's concept (not the sidebar selection)
            badges = keyword_badges(eval_res)
            if badges:
                st.html(keyword_badges_html(badges))
            if badges and all(status not in (FOUND, MISSING) for _, status in badges):
                st.caption("Keyword matches were not stored for this older attempt.")

            st.markdown("---")
    
            # Transcription Display
            st.markdown("#### 📝 Speech Transcription")
            st.html(transcript_html(eval_res["transcription"]))
            st.html("<br>")
    
            # Educational AI Feedback Section
            st.markdown("#### 💡 AI Tutor Review & Recommendations")
            notice = feedback_notice(eval_res)
            if notice:
                st.info(notice)
            st.markdown(eval_res["feedback"])

            # Download PDF Report button
            st.markdown("---")
            try:
                st.download_button(
                    label="📥 Download Detailed PDF Assessment Report",
                    data=api.get_pdf(eval_res["id"]),
                    file_name=f"VBCUA_Report_{concept_title.replace(' ', '_')}_{eval_res['id']}.pdf",
                    mime="application/pdf"
                )
            except BackendError as e:
                show_error(e)
    else:
        with st.container(border=True):
            st.html("""
            <div style='text-align: center; padding: 30px 0;'>
                <h3 style='color: white;'>🎙️ Awaiting Spoken Explanation</h3>
                <p style='color: #a0aec0; font-size: 0.95rem;'>Explain a conceptual topic in your own words, and we'll analyze both your comprehension and your delivery fluency here.</p>
            </div>
            """)

# Evaluation History Timeline
st.markdown("---")
st.markdown("### 🕰️ Evaluation Performance History")
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
        hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="single-row", key="history_table",
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
    st.info("No evaluations completed yet. Record or upload an audio file to view your performance history.")
