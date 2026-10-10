import type { Evaluation } from '../types'
import { conceptTitle, formatDate, gradeTone } from '../lib/display'

/** Past attempts, newest first; selecting one opens it without reloading the page. */
export function History({ items, currentId, onOpen }: { items: Evaluation[]; currentId: number | null; onOpen: (e: Evaluation) => void }) {
  const sorted = [...items].sort((a, b) => b.id - a.id)
  if (sorted.length === 0) return <p className="muted small">No attempts yet. Your past attempts will appear here.</p>
  return (
    <ul className="history">
      {sorted.map((e) => (
        <li key={e.id}>
          <button type="button" className="history-item" aria-current={e.id === currentId} onClick={() => onOpen(e)}>
            <div style={{ minWidth: 0 }}>
              <div className="history-title">{conceptTitle(e)}</div>
              <div className="history-meta">
                #{e.id} · {formatDate(e.created_at)} · {Math.round(e.speech_duration)} s ·{' '}
                <span className={`badge-${gradeTone(e.understanding_grade)}`} style={{ background: 'none' }}>
                  {e.understanding_grade.replace(' Understanding', '')}
                </span>
              </div>
            </div>
            <div className="bar" aria-hidden><span style={{ width: `${Math.max(0, Math.min(100, e.final_score))}%` }} /></div>
            <div className="history-score">{e.final_score.toFixed(1)}</div>
          </button>
        </li>
      ))}
    </ul>
  )
}
