import { describe, expect, it } from 'vitest'
import { attempt, CLOUD } from '../test/fixtures'
import {
  conceptTitle, demoteHeadings, detailsRows, feedbackInProgress, feedbackNotice, fmt, formatClock,
  gradeTone, initialConceptId, keywordBadges, randomConcept, scoreCards,
} from './display'

describe('formatting', () => {
  it('formats numbers and missing values', () => {
    expect(fmt(76.94)).toBe('76.9')
    expect(fmt(null)).toBe('n/a')
    expect(fmt(25, 0, '%')).toBe('25%')
    expect(formatClock(65)).toBe('1:05')
  })
  it('titles come from the evaluated concept, never a selection', () => {
    expect(conceptTitle(attempt())).toBe('Cloud Computing')
    expect(conceptTitle(attempt({ concept: null, concept_id: 9 }))).toBe('Concept #9')
  })
  it('grade tone', () => {
    expect(gradeTone('Strong Understanding')).toBe('strong')
    expect(gradeTone('Moderate Understanding')).toBe('moderate')
    expect(gradeTone('Poor Understanding')).toBe('poor')
  })
})

describe('keywordBadges', () => {
  it('uses the stored lists', () => {
    expect(keywordBadges(attempt())).toEqual([
      { keyword: 'servers', status: 'found' },
      { keyword: 'storage', status: 'missing' },
    ])
  })
  it('shows the evaluated concept keywords as not recorded for older attempts', () => {
    const badges = keywordBadges(attempt({ mentioned_keywords: null, missing_keywords: null }))
    expect(badges.map((b) => b.status)).toEqual(['not-recorded', 'not-recorded'])
  })
})

describe('scoreCards and details', () => {
  it('uses stored values and the v1 pause ratio', () => {
    const [overall, understanding, fluency] = scoreCards(attempt())
    expect(overall.value).toBeCloseTo(76.94)
    expect(understanding.caption).toBe('Similarity 85.0 · keywords 25%')
    expect(fluency.caption).toBe('Pauses 26% of speaking time')
  })
  it('older attempts show n/a and fall back to the B0 pause ratio', () => {
    const [, understanding, fluency] = scoreCards(
      attempt({ understanding_score: null, keyword_coverage: null, fluency_score: null, pause_ratio_v1: null, pause_ratio: 0.4 }),
    )
    expect(understanding.value).toBeNull()
    expect(understanding.caption).toContain('keywords n/a')
    expect(fluency.caption).toBe('Pauses 40% of speaking time')
  })
  it('details list every stored sub-score', () => {
    const rows = Object.fromEntries(
      detailsRows(attempt({ asr_device: 'cpu-fp32', feedback_seconds: 4.71 })).map((r) => [r.measure, r.value]),
    )
    expect(rows['Cosine similarity']).toBe('0.850')
    expect(rows['Pause ratio']).toBe('26.3%')
    expect(rows['Speech-to-text device']).toBe('cpu-fp32')
    expect(rows['Feedback time']).toBe('4.7 s')
  })
})

describe('feedback state', () => {
  it('in progress only while AI feedback is pending or generating', () => {
    expect(feedbackInProgress(attempt({ feedback_source: 'rule-based', feedback_status: 'pending' }))).toBe(true)
    expect(feedbackInProgress(attempt({ feedback_source: 'rule-based', feedback_status: 'generating' }))).toBe(true)
    expect(feedbackInProgress(attempt({ feedback_source: 'rule-based', feedback_status: 'failed' }))).toBe(false)
    expect(feedbackInProgress(attempt({ feedback_source: 'llm', feedback_status: 'pending' }))).toBe(false)
  })
  it('explains rule-based feedback', () => {
    expect(feedbackNotice(attempt())).toBeNull()
    expect(feedbackNotice(attempt({ feedback_source: 'rule-based', feedback_status: 'pending' }))).toBeNull()
    expect(feedbackNotice(attempt({ feedback_source: 'rule-based', feedback_status: 'failed' }))).toContain('could not be generated')
    expect(feedbackNotice(attempt({ feedback_source: 'rule-based', feedback_status: 'unavailable' }))).toContain('No AI key')
  })
  it('demotes headings', () => {
    expect(demoteHeadings('# A\n## B\n#### C\nText # x\n#tag')).toBe('#### A\n#### B\n#### C\nText # x\n#tag')
  })
})

describe('concept selection', () => {
  const list = [{ ...CLOUD, id: 3 }, { ...CLOUD, id: 5 }]
  it('restores ?concept from the URL when it exists', () => {
    expect(initialConceptId(list, '5')).toBe(5)
    expect(initialConceptId(list, '9')).toBe(3)
    expect(initialConceptId(list, 'abc')).toBe(3)
    expect(initialConceptId(list, null)).toBe(3)
    expect(initialConceptId([], '1')).toBeNull()
  })
  it('surprise me never repeats the current concept when there is a choice', () => {
    for (const r of [0, 0.5, 0.99]) expect(randomConcept(list, 3, () => r)?.id).toBe(5)
    expect(randomConcept([list[0]], 3)?.id).toBe(3)
    expect(randomConcept([], null)).toBeNull()
  })
})

describe('AI-generated topics', () => {
  const ai = { ...CLOUD, id: 4, title: 'Baahubali Movie Overview', source: 'ai-generated' as const }
  const lib = { ...CLOUD, id: 2, source: 'library' as const }
  const custom = { ...CLOUD, id: 6, title: 'REST API', source: 'custom' as const }

  it('are never the default selection', () => {
    expect(initialConceptId([ai, lib, custom], null)).toBe(2)
    expect(initialConceptId([ai, custom], null)).toBe(6)
  })
  it('are still opened when the URL names them', () => {
    expect(initialConceptId([ai, lib], '4')).toBe(4)
  })
  it('are not picked by Surprise me', () => {
    for (const r of [0, 0.5, 0.99]) expect(randomConcept([ai, lib, custom], 2, () => r)?.id).toBe(6)
  })
})
