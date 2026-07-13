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

def evaluate_semantic_understanding(
    user_transcript: str, 
    reference_description: str, 
    keywords: List[str]
) -> Dict[str, Any]:
    """
    Computes semantic similarity using Sentence-BERT and verifies keyword coverage.
    """
    if not user_transcript.strip():
        return {
            "similarity_score": 0.0,
            "keyword_coverage": 0.0,
            "mentioned_keywords": [],
            "missing_keywords": keywords,
            "combined_score": 0.0
        }

    # 1. Cosine Similarity via S-BERT
    model = get_sbert_model()
    embeddings = model.encode([user_transcript, reference_description])
    
    cosine_sim = calculate_cosine_similarity(embeddings[0], embeddings[1])
    # Map cosine similarity [-1, 1] to [0, 100]
    similarity_score = max(0.0, min(100.0, (cosine_sim + 1.0) / 2.0 * 100.0))

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
        "keyword_coverage": round(keyword_coverage, 1),
        "mentioned_keywords": mentioned_keywords,
        "missing_keywords": missing_keywords,
        "combined_score": round(combined_score, 1)
    }
