import os
import re
import requests
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Load local environment variables (for Gemini key check)
load_dotenv()

# API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

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
def get_all_concepts():
    try:
        response = requests.get(f"{BACKEND_URL}/api/concepts", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

def create_new_concept(title, description, keywords):
    try:
        payload = {
            "title": title,
            "description": description,
            "keywords": [kw.strip() for kw in keywords.split(",") if kw.strip()]
        }
        response = requests.post(f"{BACKEND_URL}/api/concepts", json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error creating concept: {e}")
        return False

def run_evaluation(concept_id, audio_bytes, filename="audio.wav"):
    try:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"concept_id": concept_id}
        response = requests.post(f"{BACKEND_URL}/api/evaluate", files=files, data=data, timeout=120)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, response.json().get("detail", "Evaluation failed")
    except Exception as e:
        return None, str(e)

def get_evaluations_history():
    try:
        response = requests.get(f"{BACKEND_URL}/api/evaluations", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# Sidebar configuration
with st.sidebar:
    st.markdown("### 📚 Practice Concept")
    
    concepts = get_all_concepts()
    if concepts:
        concept_titles = [c["title"] for c in concepts]
        selected_title = st.selectbox("Choose a concept to explain:", concept_titles)
        selected_concept = next(c for c in concepts if c["title"] == selected_title)
        
        # Show Concept Details
        st.markdown("**📖 Description**")
        st.html(f"<p style='color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 20px;'>{selected_concept['description']}</p>")
        
        # Show Keywords
        st.markdown("**🔑 Required Keywords**")
        kw_html = ""
        for kw in selected_concept["keywords"]:
            kw_html += f"<span class='badge badge-indigo'>{kw}</span>"
        st.html(kw_html)
    else:
        st.info("Connecting to backend database... Please make sure backend is running.")
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
                if create_new_concept(new_title, new_desc, new_kws):
                    st.success("Concept created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create concept.")
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
            if st.button("🚀 Analyze Explanation"):
                with st.spinner("Processing audio... Speech-to-text transcription, Librosa audio feature extraction, and Sentence-BERT semantic analysis are underway..."):
                    result, err = run_evaluation(selected_concept["id"], audio_data, filename)
                    if err:
                        st.error(f"Analysis failed: {err}")
                    else:
                        st.session_state.current_evaluation = result
                        st.success("Analysis complete!")
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
        concept_title = eval_res["concept"]["title"] if eval_res.get("concept") else selected_title
        
        with st.container(border=True):
            st.markdown(f"### 📊 Analysis Results: **{concept_title}**")
            
            # Dashboard KPIs
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.html(f"""
                <div class='metric-box purple'>
                    <div class='metric-label'>Overall Score</div>
                    <div class='metric-value purple'>{eval_res['final_score']:.1f}%</div>
                    <div style='font-size:0.8rem; color:#bd00ff;'>{eval_res['understanding_grade']}</div>
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
                # Simple fluency display
                fluency_display = 100.0 - (eval_res["filler_words_ratio"] * 4.0)
                fluency_display = max(10.0, min(100.0, fluency_display))
                pause_display = eval_res.get("pause_ratio_v1")
                if pause_display is None:
                    pause_display = eval_res["pause_ratio"]
                st.html(f"""
                <div class='glass-card' style='padding:15px !important; margin:0 !important; border-color: rgba(0, 255, 102, 0.2) !important; text-align:center;'>
                    <div class='metric-label'>Speech Fluency</div>
                    <div class='metric-value green'>{fluency_display:.0f}%</div>
                    <div style='font-size:0.8rem; color:#00ff66;'>P: {pause_display*100:.0f}% | Fillers (exp.): {eval_res['filler_words_count']}</div>
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
            if selected_concept:
                trans_clean = eval_res["transcription"].lower()
                kws = selected_concept["keywords"]
                
                kw_elements = ""
                for kw in kws:
                    pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                    if re.search(pattern, trans_clean) or kw.lower() in trans_clean:
                        kw_elements += f"<span class='badge badge-success'>✓ {kw}</span>"
                    else:
                        kw_elements += f"<span class='badge badge-danger'>✗ {kw}</span>"
                st.html(kw_elements)
                
            st.markdown("---")
    
            # Transcription Display
            st.markdown("#### 📝 Speech Transcription")
            st.html(f"""
            <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; font-style: italic; color: #a0aec0;'>
                "{eval_res['transcription']}"
            </div>
            """)
            st.html("<br>")
    
            # Educational AI Feedback Section
            st.markdown("#### 💡 AI Tutor Review & Recommendations")
            st.markdown(eval_res["feedback"])
            
            # Download PDF Report button
            st.markdown("---")
            try:
                pdf_url = f"{BACKEND_URL}/api/evaluations/{eval_res['id']}/pdf"
                pdf_response = requests.get(pdf_url, timeout=10)
                if pdf_response.status_code == 200:
                    st.download_button(
                        label="📥 Download Detailed PDF Assessment Report",
                        data=pdf_response.content,
                        file_name=f"VBCUA_Report_{concept_title.replace(' ', '_')}_{eval_res['id']}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("Failed to prepare PDF download file.")
            except Exception as e:
                st.error(f"Error fetching report: {e}")
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
history = get_evaluations_history()

if history:
    history_data = []
    for item in history:
        concept_name = item["concept"]["title"] if item.get("concept") else f"Concept ID {item['concept_id']}"
        created_time = datetime.fromisoformat(item["created_at"].replace("Z", "")).strftime("%b %d, %Y %H:%M")
        
        # Map grades to colors
        grade = item["understanding_grade"]
        if "Strong" in grade:
            grade_styled = f"<span style='color: #00ff66; font-weight: bold;'>{grade}</span>"
        elif "Moderate" in grade:
            grade_styled = f"<span style='color: #ffe600; font-weight: bold;'>{grade}</span>"
        else:
            grade_styled = f"<span style='color: #ff416c; font-weight: bold;'>{grade}</span>"
            
        history_data.append({
            "Date": created_time,
            "Concept": concept_name,
            "Score": f"{item['final_score']:.1f}%",
            "Grade": grade_styled,
            "Duration": f"{item['speech_duration']:.1f}s",
            "Fillers": item["filler_words_count"],
            "Action": item
        })
        
    # Check if a specific attempt was requested via query parameter View click
    query_params = st.query_params
    if "load_attempt" in query_params:
        try:
            attempt_idx = int(query_params["load_attempt"])
            if 0 <= attempt_idx < len(history_data):
                st.session_state.current_evaluation = history_data[attempt_idx]["Action"]
                st.query_params.clear()
                st.rerun()
        except Exception:
            pass
        
    # Render custom timeline table
    import textwrap
    html_table = """
    <table style='width:100%; border-collapse: collapse; margin-top:15px; color:#f1f3f9;'>
        <thead>
            <tr style='border-bottom: 2px solid rgba(255,255,255,0.1); text-align: left;'>
                <th style='padding: 10px;'>Date</th>
                <th style='padding: 10px;'>Concept</th>
                <th style='padding: 10px;'>Overall Score</th>
                <th style='padding: 10px;'>Grade</th>
                <th style='padding: 10px;'>Duration</th>
                <th style='padding: 10px;'>Filler Words (experimental)</th>
                <th style='padding: 10px;'>Inspect</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, row in enumerate(history_data):
        border_style = "border-bottom: 1px solid rgba(255,255,255,0.05);"
        html_table += f"""
            <tr style='{border_style}'>
                <td style='padding: 10px;'>{row['Date']}</td>
                <td style='padding: 10px; font-weight: 600;'>{row['Concept']}</td>
                <td style='padding: 10px;'>{row['Score']}</td>
                <td style='padding: 10px;'>{row['Grade']}</td>
                <td style='padding: 10px;'>{row['Duration']}</td>
                <td style='padding: 10px;'>{row['Fillers']}</td>
                <td style='padding: 10px;'><a href='/?load_attempt={i}' target='_self' style='color: #bd00ff; text-decoration: none; font-weight: bold;'>View</a></td>
            </tr>
        """
    html_table += "</tbody></table>"
    st.html(html_table)
    
    # Inspection selection dropdown
    st.write("")
    col_sel, col_btn = st.columns([8, 2])
    with col_sel:
        selected_attempt_idx = st.selectbox(
            "Select a past attempt to inspect:",
            options=range(len(history_data)),
            format_func=lambda idx: f"Attempt #{len(history_data)-idx}: {history_data[idx]['Concept']} - {history_data[idx]['Date']} (Score: {history_data[idx]['Score']})"
        )
    with col_btn:
        st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True) # spacing
        if st.button("🔍 Load Attempt"):
            st.session_state.current_evaluation = history_data[selected_attempt_idx]["Action"]
            st.rerun()
else:
    st.info("No evaluations completed yet. Record or upload an audio file to view your performance history.")
