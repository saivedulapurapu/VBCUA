export interface Concept {
  id: number
  title: string
  description: string
  keywords: string[]
  /** "library" (curated), "custom" (added by a user) or "ai-generated"; null for older rows */
  source?: 'library' | 'custom' | 'ai-generated' | null
  created_at?: string
}

export interface TrendingTopic {
  title: string
  category: string
  why: string
}

export interface TrendingResponse {
  date: string
  /** "google-search" = live trending topics; "curated" = fallback list, never shown as trending */
  origin: 'google-search' | 'curated'
  topics: TrendingTopic[]
  sources: Array<{ title: string; uri: string }>
  /** Google's search-suggestions HTML, which must be displayed with grounded results */
  search_entry_point: string | null
}

export type FeedbackStatus = 'pending' | 'generating' | 'done' | 'failed' | 'unavailable'

/** One stored attempt, as returned by the backend (older rows have nulls for newer fields). */
export interface Evaluation {
  id: number
  concept_id: number
  concept?: Concept | null
  transcription: string
  asr_model?: string | null
  asr_confidence?: number | null
  asr_device?: string | null
  semantic_score: number
  cosine_similarity?: number | null
  keyword_coverage?: number | null
  mentioned_keywords?: string[] | null
  missing_keywords?: string[] | null
  understanding_score?: number | null
  semantic_method?: string | null
  fluency_score?: number | null
  fluency_method?: string | null
  filler_words_count: number
  filler_words_ratio: number
  pause_ratio: number
  pause_ratio_v1?: number | null
  rms_energy: number
  speech_duration: number
  /** Delivery measures: stored for later analysis, not scored */
  speaking_rate_wpm?: number | null
  long_pause_count?: number | null
  long_pauses_per_min?: number | null
  longest_pause_s?: number | null
  final_score: number
  understanding_grade: string
  feedback: string
  feedback_source?: 'llm' | 'rule-based' | null
  llm_model?: string | null
  feedback_status?: FeedbackStatus | null
  feedback_seconds?: number | null
  recording_warning?: string | null
  created_at: string
}
