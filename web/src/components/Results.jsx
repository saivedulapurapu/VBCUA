import { useId, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Download, Info, MicOff, RefreshCw, Repeat } from 'lucide-react'
import { api } from '../api'

import {
  conceptTitle, deliveryRows, demoteHeadings, detailsRows, feedbackInProgress, feedbackNotice, formatDate,
  gradeTone, keywordBadges, scoreCards,
} from '../lib/display'
import { ScoreRing } from './ScoreRing'
import { ErrorNotice } from './ErrorNotice'

const TABS = ['Feedback', 'Keywords', 'Transcript', 'Details'] 











/** Step 3: scores, notices and tabs for one attempt. All values come from the stored attempt. */
export function Results({ evaluation: e, feedbackError, onRetryFeedback, canPractise = false, onPractise }) {
  const [tab, setTab] = useState('Feedback')
  const tabIds = useId()
  const tabRefs = useRef([])
  const title = conceptTitle(e)
  const notice = feedbackNotice(e)
  const writing = feedbackInProgress(e)
  const badges = keywordBadges(e)
  const found = badges.filter((b) => b.status === 'found').length
  const recorded = badges.some((b) => b.status === 'found' || b.status === 'missing')

  const onTabKey = (event, index) => {
    const delta = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
    if (!delta) return
    const next = (index + delta + TABS.length) % TABS.length
    setTab(TABS[next])
    tabRefs.current[next]?.focus()
  }

  return (
    <div>
      <h2 className="step-head"><span className="step-num">3</span> Results</h2>
      <div className="result-head">
        <div>
          <h3 className="result-title">{title}</h3>
          <div className="small muted">Attempt #{e.id} · {formatDate(e.created_at)}</div>
        </div>
        <span className={`badge badge-${gradeTone(e.understanding_grade)}`}>{e.understanding_grade}</span>
      </div>

      <div className="scores">
        {scoreCards(e).map((card) => (
          <div className="score" key={card.label}>
            <ScoreRing value={card.value} label={card.label} />
            <div className="score-label">
              {card.label}
              <span title={card.help} aria-label={card.help} style={{ display: 'inline-flex', cursor: 'help' }}>
                <Info size={14} aria-hidden />
              </span>
            </div>
            <div className="score-caption">{card.caption}</div>
          </div>
        ))}
      </div>

      {e.recording_warning && (
        <div className="notice notice-warn" role="status"><MicOff size={18} aria-hidden /> {e.recording_warning}</div>
      )}
      {notice && <div className="notice notice-info" role="status"><Info size={18} aria-hidden /> {notice}</div>}

      <div className="tabs" role="tablist" aria-label="Result details">
        {TABS.map((name, i) => (
          <button
            key={name}
            ref={(el) => { tabRefs.current[i] = el }}
            type="button"
            role="tab"
            id={`${tabIds}-tab-${name}`}
            aria-selected={tab === name}
            aria-controls={`${tabIds}-panel-${name}`}
            tabIndex={tab === name ? 0 : -1}
            className="tab"
            onClick={() => setTab(name)}
            onKeyDown={(event) => onTabKey(event, i)}
          >
            {name}
          </button>
        ))}
      </div>

      <div className="tab-panel" role="tabpanel" id={`${tabIds}-panel-${tab}`} aria-labelledby={`${tabIds}-tab-${tab}`} key={tab}>
        {tab === 'Feedback' && (
          writing ? (
            <div className="progress" role="status" aria-live="polite" style={{ marginTop: 0 }}>
              <div className="spinner" aria-hidden />
              <div>
                <div style={{ fontWeight: 600 }}>Writing feedback…</div>
                <div className="small muted">Your scores are ready; AI feedback usually takes a few seconds.</div>
              </div>
            </div>
          ) : (
            <>
              {/* Markdown is rendered without raw HTML, so feedback text can never inject markup */}
              <div className="markdown"><ReactMarkdown>{demoteHeadings(e.feedback)}</ReactMarkdown></div>
              {e.feedback_status === 'failed' && (
                <button type="button" className="btn" style={{ marginTop: 12 }} onClick={onRetryFeedback}>
                  <RefreshCw size={16} aria-hidden /> Try AI feedback again
                </button>
              )}
            </>
          )
        )}
        {feedbackError ? <ErrorNotice error={feedbackError} /> : null}

        {tab === 'Keywords' && (
          <>
            <p className="small muted" style={{ marginTop: 0 }}>
              {recorded ? `${found} of ${badges.length} key terms mentioned (whole-word matches).` : 'Keyword matches were not stored for this older attempt.'}
            </p>
            <div className="chips">
              {badges.map((b) => (
                <span key={`${b.status}-${b.keyword}`} className={`chip chip-${b.status === 'found' ? 'found' : b.status === 'missing' ? 'missing' : 'plain'}`}>
                  {b.status === 'found' ? '✓ ' : b.status === 'missing' ? '✗ ' : ''}{b.keyword}
                </span>
              ))}
            </div>
          </>
        )}

        {tab === 'Transcript' && (
          <>
            <div className="transcript">{e.transcription}</div>
            <p className="small muted">Transcribed by {e.asr_model || 'an unknown model'}. Filler words such as “um” are usually not transcribed.</p>
          </>
        )}

        {tab === 'Details' && (
          <>
            <table className="table">
              <thead><tr><th>Measure</th><th>Value</th><th>Note</th></tr></thead>
              <tbody>
                {detailsRows(e).map((row) => (
                  <tr key={row.measure}><td>{row.measure}</td><td>{row.value}</td><td className="muted">{row.note}</td></tr>
                ))}
              </tbody>
            </table>
            <h4 className="table-heading">Delivery measures (not scored)</h4>
            <table className="table" aria-label="Delivery measures (not scored)">
              <tbody>
                {deliveryRows(e).map((row) => (
                  <tr key={row.measure}><td>{row.measure}</td><td>{row.value}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      <div className="actions">
        <a className="btn" href={api.pdfUrl(e.id)} download>
          <Download size={16} aria-hidden /> Download PDF report
        </a>
        {canPractise && onPractise && (
          <button type="button" className="btn" onClick={onPractise}>
            <Repeat size={16} aria-hidden /> Practise this concept again
          </button>
        )}
      </div>
    </div>
  )
}
