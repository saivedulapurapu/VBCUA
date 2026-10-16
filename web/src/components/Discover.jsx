import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, ExternalLink, Flame, Lightbulb } from 'lucide-react'
import { api } from '../api'

import { ErrorNotice } from './ErrorNotice'







const categoryClass = (category) => `cat-${category.toLowerCase().replace(/[^a-z]/g, '')}`

/**
 * "Trending today" (live from Google Search via the backend) or "Suggested topics" (curated fallback),
 * plus "Any topic": type anything and the backend prepares an AI-written reference for it.
 */
export function Discover({ concepts, onConcept }) {
  const [trending, setTrending] = useState(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const [topic, setTopic] = useState('')
  const [preparing, setPreparing] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.getTrending().then(setTrending).catch(() => setLoadFailed(true))
  }, [])

  const live = trending?.origin === 'google-search'

  // Suggestions while typing: library concepts and today's topics that match
  const suggestions = useMemo(() => {
    const q = topic.trim().toLowerCase()
    if (q.length < 2) return []
    const fromLibrary = concepts.filter((c) => c.title.toLowerCase().includes(q)).map((c) => c.title)
    const fromTrending = (trending?.topics ?? []).map((t) => t.title).filter((t) => t.toLowerCase().includes(q))
    return [...new Set([...fromLibrary, ...fromTrending])].slice(0, 5)
  }, [topic, concepts, trending])

  const prepare = async (rawTopic) => {
    const wanted = rawTopic.trim()
    if (wanted.length < 2 || preparing) return
    const existing = concepts.find((c) => c.title.toLowerCase() === wanted.toLowerCase())
    if (existing) {
      onConcept(existing)
      setTopic('')
      return
    }
    setPreparing(wanted)
    setError(null)
    try {
      onConcept(await api.generateConcept(wanted))
      setTopic('')
    } catch (e) {
      setError(e)
    } finally {
      setPreparing(null)
    }
  }

  return (
    <div className="discover">
      {!loadFailed && (
        <>
          <div className="discover-head">
            {live ? <Flame size={16} aria-hidden /> : <Lightbulb size={16} aria-hidden />}
            <span>{trending ? (live ? 'Trending today' : 'Suggested topics') : 'Loading topics…'}</span>
            {live && <span className="live-tag">Live from Google Search</span>}
          </div>
          <div className="chips topic-chips" aria-busy={!trending}>
            {trending
              ? trending.topics.map((t) => (
                  <button
                    key={t.title}
                    type="button"
                    className={`chip topic-chip ${categoryClass(t.category)}`}
                    title={t.why || t.category}
                    onClick={() => void prepare(t.title)}
                    disabled={preparing !== null}
                  >
                    <span className="cat-dot" aria-hidden />
                    {t.title}
                  </button>
                ))
              : Array.from({ length: 6 }, (_, i) => <span key={i} className="chip skeleton" aria-hidden />)}
          </div>
        </>
      )}

      <form
        className="any-topic"
        onSubmit={(e) => {
          e.preventDefault()
          void prepare(topic)
        }}
      >
        <label className="sr-only" htmlFor="any-topic">Any topic</label>
        <input
          id="any-topic"
          className="field"
          placeholder="Or type any topic, e.g. black holes, UPI payments…"
          value={topic}
          maxLength={120}
          onChange={(e) => setTopic(e.target.value)}
          disabled={preparing !== null}
          list="any-topic-suggestions"
          autoComplete="off"
        />
        <datalist id="any-topic-suggestions">
          {suggestions.map((s) => <option key={s} value={s} />)}
        </datalist>
        <button type="submit" className="btn" disabled={topic.trim().length < 2 || preparing !== null} aria-label="Use this topic">
          <ArrowRight size={16} aria-hidden />
        </button>
      </form>

      {preparing && (
        <div className="progress" role="status" aria-live="polite" style={{ marginTop: 10 }}>
          <div className="spinner" aria-hidden />
          <div className="small">Preparing a reference for “{preparing}”…</div>
        </div>
      )}
      {error ? <ErrorNotice error={error} /> : null}

      {live && trending && (
        <>
          {trending.search_entry_point && (
            // Google requires showing its search suggestions with grounded results. The HTML comes
            // from Google; it is isolated in a sandboxed frame (no scripts, links open in a new tab).
            <iframe
              className="search-suggestions"
              title="Google Search suggestions"
              sandbox="allow-popups allow-popups-to-escape-sandbox"
              srcDoc={`<base target="_blank">${trending.search_entry_point}`}
            />
          )}
          {trending.sources.length > 0 && (
            <details className="disclosure">
              <summary>Sources ({trending.sources.length})</summary>
              <ul className="sources">
                {trending.sources.map((s) => (
                  <li key={s.uri}>
                    <a href={s.uri} target="_blank" rel="noopener noreferrer">
                      {s.title} <ExternalLink size={12} aria-hidden />
                    </a>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </div>
  )
}
