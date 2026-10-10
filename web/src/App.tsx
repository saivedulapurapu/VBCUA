import { useCallback, useEffect, useRef, useState } from 'react'
import { History as HistoryIcon, Mic, Plus, Sparkles } from 'lucide-react'
import { api } from './api'
import type { Concept, Evaluation } from './types'
import { initialConceptId } from './lib/display'
import { useTheme } from './hooks/useTheme'
import { useUrlParam } from './hooks/useUrlParam'
import { useElapsed } from './hooks/useElapsed'
import { useFeedback } from './hooks/useFeedback'
import { ThemeToggle } from './components/ThemeToggle'
import { ConceptPicker } from './components/ConceptPicker'
import { AddConceptDrawer } from './components/AddConceptDrawer'
import { ErrorNotice } from './components/ErrorNotice'
import { Recorder, type AudioInput } from './components/Recorder'
import { AnalyseProgress } from './components/AnalyseProgress'
import { Results } from './components/Results'
import { History } from './components/History'

export default function App() {
  const { theme, cycle } = useTheme()
  const [conceptParam, setConceptParam] = useUrlParam('concept')
  const [attemptParam, setAttemptParam] = useUrlParam('attempt')
  const [concepts, setConcepts] = useState<Concept[]>([])
  const [history, setHistory] = useState<Evaluation[]>([])
  const [loadError, setLoadError] = useState<unknown>(null)
  const [addingConcept, setAddingConcept] = useState(false)
  const [audio, setAudio] = useState<AudioInput | null>(null)
  const [analysing, setAnalysing] = useState(false)
  const [analyseError, setAnalyseError] = useState<unknown>(null)
  const [current, setCurrent] = useState<Evaluation | null>(null)
  const elapsed = useElapsed(analysing)
  const resultsRef = useRef<HTMLElement>(null)
  const onAudioChange = useCallback((a: AudioInput | null) => setAudio(a), [])

  // Initial data; reopen the attempt from the URL so a reload keeps it
  useEffect(() => {
    api.getConcepts().then(setConcepts).catch(setLoadError)
    api.getEvaluations().then(setHistory).catch(setLoadError)
    const id = attemptParam && /^\d+$/.test(attemptParam) ? Number(attemptParam) : null
    if (id !== null) {
      api.getEvaluation(id).then(setCurrent).catch(() => setAttemptParam(null))
    }
  }, []) // run once on load

  const selectedId = initialConceptId(concepts, conceptParam)
  const selectedConcept = concepts.find((c) => c.id === selectedId) ?? null
  const selectConcept = (id: number) => setConceptParam(String(id))

  /** Shows an attempt and keeps the history list in sync with it. */
  const showEvaluation = useCallback(
    (e: Evaluation) => {
      setCurrent(e)
      setAttemptParam(String(e.id))
      setHistory((list) => (list.some((x) => x.id === e.id) ? list.map((x) => (x.id === e.id ? e : x)) : [e, ...list]))
    },
    [setAttemptParam],
  )

  const { error: feedbackError } = useFeedback(current, showEvaluation)

  const analyse = async () => {
    if (!audio || !selectedConcept) return
    setAnalysing(true)
    setAnalyseError(null)
    try {
      showEvaluation(await api.evaluate(selectedConcept.id, audio.blob, audio.filename))
      if (window.matchMedia?.('(max-width: 920px)').matches) resultsRef.current?.scrollIntoView({ behavior: 'smooth' })
    } catch (e) {
      setAnalyseError(e)
    } finally {
      setAnalysing(false)
    }
  }

  const retryFeedback = () => current && setCurrent({ ...current, feedback_status: 'pending' })

  return (
    <>
      <div className="backdrop" aria-hidden>
        <div className="blob blob-1" />
        <div className="blob blob-2" />
        <div className="blob blob-3" />
      </div>

      <div className="shell">
        <header className="topbar glass">
          <div className="brand">
            <img src="/favicon.svg" alt="" />
            <div>
              <div className="brand-name">VBCUA</div>
              <div className="brand-sub">Concept Understanding Analyser</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" className="btn" onClick={() => setAddingConcept(true)}>
              <Plus size={16} aria-hidden /> Add concept
            </button>
            <ThemeToggle theme={theme} onCycle={cycle} />
          </div>
        </header>

        <section className="hero">
          <h1>
            Explain it out loud. <span className="gradient-text">See how well you understand it.</span>
          </h1>
          <p>Pick a computing concept, explain it in your own words, and get scores and feedback on your understanding and delivery.</p>
        </section>

        {loadError ? <ErrorNotice error={loadError} /> : null}

        <main className="grid">
          <section className="panel glass" aria-labelledby="step-1">
            <h2 className="step-head" id="step-1">
              <span className="step-num">1</span> Choose a concept
            </h2>
            <ConceptPicker concepts={concepts} selectedId={selectedId} onSelect={selectConcept} />

            <hr className="divider" />
            <h2 className="step-head">
              <span className="step-num">2</span> Record or upload your explanation
            </h2>
            <Recorder onChange={onAudioChange} disabled={analysing} />

            <div style={{ marginTop: 18 }}>
              {/* The button names the concept so an explanation is never graded against the wrong one */}
              <button type="button" className="btn btn-primary btn-block" onClick={analyse} disabled={!audio || !selectedConcept || analysing}>
                <Sparkles size={18} aria-hidden />
                {selectedConcept ? `Analyse explanation of ${selectedConcept.title}` : 'Analyse explanation'}
              </button>
              {analysing && selectedConcept && <AnalyseProgress conceptTitle={selectedConcept.title} seconds={elapsed} />}
              {analyseError ? <ErrorNotice error={analyseError} /> : null}
            </div>
          </section>

          <section className="panel glass" aria-label="Results" ref={resultsRef}>
            {current ? (
              <Results evaluation={current} feedbackError={feedbackError} onRetryFeedback={retryFeedback} />
            ) : (
              <div className="empty">
                <Mic size={40} aria-hidden />
                <h3>No results yet</h3>
                <p>Choose a concept, record or upload your explanation, then select Analyse.</p>
              </div>
            )}
          </section>
        </main>

        <section className="panel glass section" aria-labelledby="history-title">
          <h2 className="step-head" id="history-title">
            <HistoryIcon size={18} aria-hidden /> History
          </h2>
          <History items={history} currentId={current?.id ?? null} onOpen={showEvaluation} />
        </section>
      </div>

      {addingConcept && (
        <AddConceptDrawer
          onClose={() => setAddingConcept(false)}
          onCreated={(concept) => {
            setConcepts((list) => [...list, concept])
            selectConcept(concept.id)
            setAddingConcept(false)
          }}
        />
      )}
    </>
  )
}
