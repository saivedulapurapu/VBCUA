import { formatClock } from '../lib/display'

/** Shown while the backend transcribes and scores the explanation. */
export function AnalyseProgress({ conceptTitle, seconds }) {
  return (
    <div className="progress glass" role="status" aria-live="polite">
      <div className="spinner" aria-hidden />
      <div>
        <div style={{ fontWeight: 600 }}>Analysing your explanation of {conceptTitle}…</div>
        <div className="small muted">Transcribing speech and scoring understanding and fluency. Feedback follows separately.</div>
        <div className="small muted" style={{ marginTop: 4 }}>Elapsed: {formatClock(seconds)}</div>
      </div>
    </div>
  )
}
