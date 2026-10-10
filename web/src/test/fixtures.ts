import type { Concept, Evaluation } from '../types'

export const CLOUD: Concept = { id: 2, title: 'Cloud Computing', description: 'Cloud def', keywords: ['servers', 'storage'] }

export const CONCEPTS: Concept[] = [
  { id: 1, title: 'Machine Learning', description: 'ML def', keywords: ['training'] },
  CLOUD,
  { id: 3, title: 'Database Management System (DBMS)', description: 'DB def', keywords: ['sql', 'tables'] },
]

export function attempt(overrides: Partial<Evaluation> = {}): Evaluation {
  return {
    id: 7,
    concept_id: 2,
    concept: CLOUD,
    transcription: 'We rent servers.',
    asr_model: 'whisper-small',
    asr_confidence: -0.22,
    semantic_score: 85,
    cosine_similarity: 0.8502,
    keyword_coverage: 25,
    mentioned_keywords: ['servers'],
    missing_keywords: ['storage'],
    understanding_score: 67,
    fluency_score: 100,
    filler_words_count: 0,
    filler_words_ratio: 0,
    pause_ratio: 0.5,
    pause_ratio_v1: 0.263,
    rms_energy: 0.05,
    speech_duration: 81.9,
    final_score: 76.94,
    understanding_grade: 'Moderate Understanding',
    feedback: '#### Suggestions\nGood work.',
    feedback_source: 'llm',
    llm_model: 'gemini-3.5-flash-lite',
    feedback_status: 'done',
    created_at: '2026-10-10T20:10:00.123',
    ...overrides,
  }
}
