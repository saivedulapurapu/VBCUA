/** Formatting and derived values for an attempt. Only reads stored backend values; never re-scores. */

 

export function fmt(value, digits = 1, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  return `${value.toFixed(digits)}${suffix}`
}

export function conceptTitle(e) {
  return e.concept?.title || `Concept #${e.concept_id}`
}

export function pauseRatio(e) {
  return e.pause_ratio_v1 ?? e.pause_ratio ?? null
}

export function keywordBadges(e) {
  if (e.mentioned_keywords && e.missing_keywords) {
    return [
      ...e.mentioned_keywords.map((keyword) => ({ keyword, status: 'found'  })),
      ...e.missing_keywords.map((keyword) => ({ keyword, status: 'missing'  })),
    ]
  }
  return (e.concept?.keywords ?? []).map((keyword) => ({ keyword, status: 'not-recorded'  }))
}

 

export function gradeTone(grade) {
  if (grade.includes('Strong')) return 'strong'
  if (grade.includes('Moderate')) return 'moderate'
  return 'poor'
}








export function scoreCards(e) {
  const pause = pauseRatio(e)
  return [
    {
      label: 'Overall',
      value: e.final_score ?? null,
      caption: '70% understanding + 30% fluency',
      help: 'Weighted combination of the understanding and fluency scores (0–100).',
    },
    {
      label: 'Understanding',
      value: e.understanding_score ?? null,
      caption: `Similarity ${fmt(e.semantic_score)} · keywords ${fmt(e.keyword_coverage, 0, '%')}`,
      help: '70% semantic similarity to the reference definition + 30% keyword coverage.',
    },
    {
      label: 'Fluency (provisional)',
      value: e.fluency_score ?? null,
      caption: `Pauses ${fmt(pause === null ? null : pause * 100, 0, '%')} of speaking time`,
      help: 'Based on pause ratio only; calibration pending.',
    },
  ]
}

export function detailsRows(e) {
  return [
    { measure: 'Semantic similarity', value: fmt(e.semantic_score), note: 'max(0, cosine) × 100' },
    { measure: 'Cosine similarity', value: fmt(e.cosine_similarity, 3), note: 'Sentence-BERT, all-MiniLM-L6-v2' },
    { measure: 'Keyword coverage', value: fmt(e.keyword_coverage, 1, '%'), note: 'Whole-word matches' },
    { measure: 'Understanding score', value: fmt(e.understanding_score), note: '70% similarity + 30% keywords' },
    { measure: 'Pause ratio', value: fmt(e.pause_ratio_v1 == null ? null : e.pause_ratio_v1 * 100, 1, '%'), note: 'Between first and last speech' },
    { measure: 'Fluency score', value: fmt(e.fluency_score), note: 'Pause-based' },
    { measure: 'Overall score', value: fmt(e.final_score), note: '70% understanding + 30% fluency' },
    { measure: 'Filler words', value: String(e.filler_words_count ?? 'n/a'), note: 'Experimental, not scored' },
    { measure: 'Duration', value: fmt(e.speech_duration, 1, ' s'), note: '' },
    { measure: 'Speech-to-text model', value: e.asr_model || 'n/a', note: `ASR confidence ${fmt(e.asr_confidence, 3)}` },
    { measure: 'Speech-to-text device', value: e.asr_device || 'n/a', note: 'cpu-fp32 or cuda-fp16' },
    { measure: 'Feedback source', value: e.feedback_source || 'n/a', note: e.llm_model || '' },
    { measure: 'Feedback time', value: fmt(e.feedback_seconds, 1, ' s'), note: 'Time taken by the AI feedback model' },
    { measure: 'Scoring methods', value: `semantic ${e.semantic_method || 'n/a'}, fluency ${e.fluency_method || 'n/a'}`, note: 'B0 = original baseline' },
  ]
}

/** Raw delivery measures (not scored); older attempts show n/a. */
export function deliveryRows(e) {
  return [
    { measure: 'Speaking rate', value: fmt(e.speaking_rate_wpm, 0, ' words/min') },
    { measure: 'Long pauses (1 s or more)', value: e.long_pause_count == null ? 'n/a' : String(e.long_pause_count) },
    { measure: 'Long pauses per minute', value: fmt(e.long_pauses_per_min, 1) },
    { measure: 'Longest pause', value: fmt(e.longest_pause_s, 1, ' s') },
  ]
}

export function feedbackInProgress(e) {
  return e.feedback_source !== 'llm' && (e.feedback_status === 'pending' || e.feedback_status === 'generating')
}

export function feedbackNotice(e) {
  if (e.feedback_source !== 'rule-based' || feedbackInProgress(e)) return null
  if (e.feedback_status === 'failed') {
    return 'AI feedback could not be generated (the AI service was busy or unavailable), so this feedback was produced by the built-in rule-based generator.'
  }
  if (e.feedback_status === 'unavailable') {
    return 'No AI key is configured, so this feedback was produced by the built-in rule-based generator.'
  }
  return 'AI feedback was unavailable for this attempt, so this feedback was produced by the built-in rule-based generator.'
}

/** Any markdown heading (#…######) becomes a small "####" heading so it fits the results card. */
export function demoteHeadings(markdown) {
  return markdown
    .split('\n')
    .map((line) => {
      const m = /^\s*(#{1,6})\s(.*)$/.exec(line)
      return m ? `#### ${m[2]}` : line
    })
    .join('\n')
}

export function formatDate(iso) {
  return iso ? iso.slice(0, 16).replace('T', ' ') : ''
}

/** AI-generated topics are practice extras: never the default and not part of "Surprise me". */
export const isPracticeTopic = (c) => c.source === 'ai-generated'

/**
 * Concept to preselect: the URL's ?concept=<id> if it exists, else the first library or custom
 * concept (an AI-generated topic is only selected when the URL asks for it).
 */
export function initialConceptId(concepts, requested) {
  const wanted = requested !== null && /^\d+$/.test(requested) ? Number(requested) : null
  if (wanted !== null && concepts.some((c) => c.id === wanted)) return wanted
  return (concepts.find((c) => !isPracticeTopic(c)) ?? concepts[0])?.id ?? null
}

/** Picks a random library or custom concept, avoiding the current one when there is a choice. */
export function randomConcept(concepts, currentId, random = Math.random) {
  const eligible = concepts.filter((c) => !isPracticeTopic(c))
  const pool = eligible.length > 1 ? eligible.filter((c) => c.id !== currentId) : eligible
  if (pool.length === 0) return null
  return pool[Math.floor(random() * pool.length)]
}

export function formatClock(seconds) {
  const s = Math.max(0, Math.floor(seconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}
