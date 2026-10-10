import logging
import re
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Lazy load model
_sbert_model = None

def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def similarity_from_cosine_b0(cosine_sim: float) -> float:
    """
    Baseline B0 (original method): maps cosine [-1, 1] to [0, 100].
    Gives unrelated text (cos ~ 0) a score of ~50. Kept only for comparison.
    """
    return max(0.0, min(100.0, (cosine_sim + 1.0) / 2.0 * 100.0))

def similarity_from_cosine(cosine_sim: float) -> float:
    """
    Maps cosine to [0, 100] with no floor: unrelated (cos <= 0) -> 0, identical -> 100.
    """
    return max(0.0, min(100.0, cosine_sim * 100.0))

SIMILARITY_METHODS = {
    "v1": similarity_from_cosine,
    "b0": similarity_from_cosine_b0,
}

def evaluate_semantic_understanding(
    user_transcript: str,
    reference_description: str,
    keywords: List[str],
    method: str = "v1"
) -> Dict[str, Any]:
    """
    Computes semantic similarity using Sentence-BERT and verifies keyword coverage.
    method: "v1" (default, no floor) or "b0" (original (cos + 1) / 2 baseline).
    """
    if method not in SIMILARITY_METHODS:
        raise ValueError(f"Unknown similarity method: {method!r}")

    if not user_transcript.strip():
        return {
            "similarity_score": 0.0,
            "cosine_similarity": 0.0,
            "scoring_method": method,
            "keyword_coverage": 0.0,
            "mentioned_keywords": [],
            "missing_keywords": keywords,
            "combined_score": 0.0
        }

    # 1. Cosine Similarity via S-BERT
    model = get_sbert_model()
    embeddings = model.encode([user_transcript, reference_description])

    cosine_sim = calculate_cosine_similarity(embeddings[0], embeddings[1])
    similarity_score = SIMILARITY_METHODS[method](cosine_sim)

    # 2. Keyword Coverage
    mentioned_keywords = []
    missing_keywords = []
    
    clean_transcript = user_transcript.lower()
    
    for keyword in keywords:
        # Check if keyword is in the transcript as a substring
        # We can make this slightly smarter by checking boundaries
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        
        # Simple fallback to substring check if word boundaries are complex (e.g. phrases)
        if re.search(pattern, clean_transcript) or keyword.lower() in clean_transcript:
            mentioned_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
            
    total_keywords = len(keywords)
    keyword_coverage = (len(mentioned_keywords) / total_keywords * 100.0) if total_keywords > 0 else 100.0

    # 3. Combined Score
    # 70% Sentence similarity, 30% keyword coverage
    combined_score = (similarity_score * 0.7) + (keyword_coverage * 0.3)

    return {
        "similarity_score": round(similarity_score, 1),
        "cosine_similarity": round(cosine_sim, 4),
        "scoring_method": method,
        "keyword_coverage": round(keyword_coverage, 1),
        "mentioned_keywords": mentioned_keywords,
        "missing_keywords": missing_keywords,
        "combined_score": round(combined_score, 1)
    }
