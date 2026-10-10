import logging
from typing import Dict, Any, List, Optional
from backend.app.services import llm_client

logger = logging.getLogger(__name__)

FLUENCY_METHODS = ("v1", "b0")

def compute_fluency_score(
    filler_words_ratio: float,
    pause_ratio: float,
    rms_energy: Optional[float] = None,
    method: str = "v1"
) -> Dict[str, Any]:
    """
    Computes a speech fluency score out of 100 and returns observations.
    method: "v1" (default) scores pauses only; pass the trimmed pause_ratio_v1.
            Filler counts are ignored (ASR transcripts usually drop "um/uh").
            "b0" is the original formula: fillers, whole-clip pause ratio and a
            low RMS volume deduction. Kept only for comparison.
    """
    if method not in FLUENCY_METHODS:
        raise ValueError(f"Unknown fluency method: {method!r}")
    if method == "b0" and rms_energy is None:
        raise ValueError("method='b0' requires rms_energy")

    score = 100.0
    deductions = {}
    observations = []

    # 1. B0 only: Filler word deductions (ideal ratio is < 2.0 filler words per 100 words)
    if method == "b0":
        if filler_words_ratio > 2.0:
            # Deduct 4 points for every point above 2.0, capped at 30 points
            filler_deduction = min(30.0, (filler_words_ratio - 2.0) * 4.0)
            score -= filler_deduction
            deductions["filler_words"] = round(filler_deduction, 1)
            observations.append(f"Frequent use of filler words ({filler_words_ratio:.1f} per 100 words). Try replacing pauses with silence.")
        else:
            observations.append("Excellent control of filler words; speech is clear and focused.")

    # 2. Pause ratio deductions (ideal pause ratio is between 10% and 25% for natural phrasing)
    if pause_ratio > 0.30:
        # Too many pauses / hesitation
        pause_deduction = min(25.0, (pause_ratio - 0.30) * 100.0)
        score -= pause_deduction
        deductions["pause_ratio"] = round(pause_deduction, 1)
        observations.append(f"High pause ratio ({pause_ratio*100:.1f}%), indicating hesitation or slow recall.")
    elif pause_ratio < 0.05:
        # Speech is rushed
        pause_deduction = 10.0
        score -= pause_deduction
        deductions["pause_ratio"] = pause_deduction
        observations.append("Very low pause ratio; speech may feel rushed. Remember to pause at punctuation.")
    else:
        observations.append("Natural pacing and breathing intervals.")

    # 3. B0 only: RMS Energy (volume/confidence)
    # Average speech RMS is typically > 0.01. Below 0.005 is very soft/muffled.
    if method == "b0":
        if rms_energy < 0.005:
            rms_deduction = 15.0
            score -= rms_deduction
            deductions["speaking_volume"] = rms_deduction
            observations.append("Low speaking volume detected. Speak closer to the microphone or with more confidence.")
        elif rms_energy > 0.15:
            observations.append("Strong, loud speaking volume. Good delivery energy.")
        else:
            observations.append("Consistent, clear speaking volume.")

    final_score = max(10.0, min(100.0, score))
    return {
        "fluency_score": round(final_score, 1),
        "deductions": deductions,
        "observations": observations,
        "scoring_method": method
    }

def generate_rule_based_feedback(
    concept_title: str,
    semantic_result: Dict[str, Any],
    fluency_result: Dict[str, Any]
) -> str:
    """
    Rule-based backup generator for educational feedback. Used when Gemini API key is missing.
    """
    mentioned = ", ".join([f"**{kw}**" for kw in semantic_result["mentioned_keywords"]]) or "None"
    missing = ", ".join([f"**{kw}**" for kw in semantic_result["missing_keywords"]]) or "None"
    
    sim_score = semantic_result["similarity_score"]
    fluency_score = fluency_result["fluency_score"]
    
    # Concept accuracy review
    if sim_score >= 80:
        concept_review = f"You demonstrated an **excellent** understanding of {concept_title}. Your explanation aligns closely with the reference concept, showing that you grasp the core definition."
    elif sim_score >= 50:
        concept_review = f"You demonstrated a **moderate** understanding of {concept_title}. While you covered some aspects, your explanation could be more comprehensive and align closer with standard definitions."
    else:
        concept_review = f"You showed a **weak** understanding of {concept_title}. You might want to review the core definition and structure your thoughts before presenting."

    # Fluency review
    obs_bullets = "\n".join([f"- {obs}" for obs in fluency_result["observations"]])

    feedback_md = f"""### Concept Evaluation Report: **{concept_title}**

#### 1. Conceptual Accuracy & Coverage
- **Semantic Alignment**: {sim_score}%
- {concept_review}
- **Keywords Highlighted**: {mentioned}
- **Keywords Missed**: {missing}

#### 2. Speech Delivery & Fluency
- **Fluency Score**: {fluency_score}/100
{obs_bullets}

#### 3. Structured Recommendations
1. {"Review the missed keywords to make your explanation comprehensive." if semantic_result["missing_keywords"] else "Great job covering all key terms! Keep it up."}
2. {"Practice breathing and pausing naturally if you feel rushed or hesitant." if "pause_ratio" in fluency_result["deductions"] else "Maintain your current natural pacing."}
"""
    return feedback_md

def generate_ai_feedback(
    concept_title: str,
    reference_description: str,
    user_transcript: str,
    semantic_result: Dict[str, Any],
    fluency_result: Dict[str, Any]
) -> str:
    """
    Generates rich, customized feedback using the configured LLM (see llm_client).
    """
    mentioned = ", ".join(semantic_result["mentioned_keywords"])
    missing = ", ".join(semantic_result["missing_keywords"])
    observations = "; ".join(fluency_result["observations"])
    
    prompt = f"""
    You are an expert educational assessment AI. Evaluate the student's spoken explanation of the concept "{concept_title}".
    
    Here is the reference definition/description of the concept:
    ---
    {reference_description}
    ---
    
    Here is the student's transcribed explanation:
    ---
    {user_transcript}
    ---
    
    Here are some calculated statistics from the user's audio and transcript:
    - Semantic Similarity Score: {semantic_result['similarity_score']}/100
    - Key concepts mentioned: [{mentioned}]
    - Key concepts missed: [{missing}]
    - Speech analysis observations: [{observations}]
    - Fluency Score: {fluency_result['fluency_score']}/100
    
    Please write a structured evaluation report. Use markdown formatting. Include the following sections:
    1. **Conceptual Accuracy & Comprehension**: Discuss what they explained correctly, what they missed, and any misconceptions they showed.
    2. **Speech Delivery & Fluency**: Analyze their pacing and pauses based on the observations.
    3. **Key Suggestions for Improvement**: Provide 2-3 specific, actionable steps to help them explain this topic better next time.
    
    Keep the tone professional, encouraging, and constructive. Return ONLY the markdown formatted response.
    """
    
    return llm_client.generate_text(prompt)

def evaluate_explanation(
    concept_title: str,
    reference_description: str,
    user_transcript: str,
    semantic_result: Dict[str, Any],
    fluency_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main evaluation pipeline that calculates the overall grade, calls feedback generators,
    and returns the final scoring summary.
    """
    # 70% understanding, 30% speech fluency
    understanding_score = semantic_result["combined_score"]
    fluency_score = fluency_result["fluency_score"]
    
    final_score = (understanding_score * 0.7) + (fluency_score * 0.3)
    final_score = round(final_score, 1)

    # Determine Grade
    if final_score >= 80.0:
        grade = "Strong Understanding"
    elif final_score >= 50.0:
        grade = "Moderate Understanding"
    else:
        grade = "Poor Understanding"

    # Generate feedback (LLM with rule-based fallback)
    feedback = None
    feedback_source = "rule-based"
    llm_model = None
    if llm_client.is_configured():
        try:
            feedback = generate_ai_feedback(
                concept_title=concept_title,
                reference_description=reference_description,
                user_transcript=user_transcript,
                semantic_result=semantic_result,
                fluency_result=fluency_result
            )
            feedback_source = "llm"
            llm_model = llm_client.model_name()
        except Exception as e:
            logger.error(f"LLM feedback generation failed, using rule-based generator: {e}")
    if feedback is None:
        feedback = generate_rule_based_feedback(concept_title, semantic_result, fluency_result)

    return {
        "final_score": final_score,
        "understanding_grade": grade,
        "feedback": feedback,
        "feedback_source": feedback_source,
        "llm_model": llm_model
    }
