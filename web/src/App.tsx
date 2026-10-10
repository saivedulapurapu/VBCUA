import { useCallback, useEffect, useState } from 'react'
import { Mic, Plus, Sparkles } from 'lucide-react'
import { api } from './api'
import type { Concept, Evaluation } from './types'
import { initialConceptId } from './lib/display'
import { useTheme } from './hooks/useTheme'
import { useUrlParam } from './hooks/useUrlParam'
import { ThemeToggle } from './components/ThemeToggle'
import { ConceptPicker } from './components/ConceptPicker'
import { AddConceptDrawer } from './components/AddConceptDrawer'
import { ErrorNotice } from './components/ErrorNotice'
import { Recorder, type AudioInput } from './components/Recorder'
import { AnalyseProgress } from './components/AnalyseProgress'
import { useElapsed } from './hooks/useElapsed'

export default function App() {
  const { theme, cycle } = useTheme()
  const [conceptParam, setConceptParam] = useUrlParam('concept')
  const [concepts, setConcepts] = useState<Concept[]>([])
  const [loadError, setLoadError] = useState<unknown>(null)
  const [addingConcept, setAddingConcept] = useState(false)
  const [, setAttemptParam] = useUrlParam('attempt')
  const [audio, setAudio] = useState<AudioInput | null>(null)
  const [analysing, setAnalysing] = useState(false)
  const [analyseError, setAnalyseError] = useState<unknown>(null)
  const [current, setCurrent] = useState<Evaluation | null>(null)
  const elapsed = useElapsed(analysing)
  const onAudioChange = useCallback((a: AudioInput | null) => setAudio(a), [])

  useEffect(() => {
    api.getConcepts().then(setConcepts).catch(setLoadError)
  }, [])

  const selectedId = initialConceptId(concepts, conceptParam)
  const selectConcept = (id: number) => setConceptParam(String(id))
  const selectedConcept = concepts.find((c) => c.id === selectedId) ?? null

  const showEvaluation = (e: Evaluation) => {
    setCurrent(e)
    setAttemptParam(String(e.id))
  }

  const analyse = async () => {
    if (!audio || !selectedConcept) return
    setAnalysing(true)
    setAnalyseError(null)
    try {
      showEvaluation(await api.evaluate(selectedConcept.id, audio.blob, audio.filename))
    } catch (e) {
      setAnalyseError(e)
    } finally {
      setAnalysing(false)
    }
  }

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

          <section className="panel glass" aria-label="Results">
            {current ? (
              <div>
                <h2 className="step-head"><span className="step-num">3</span> Results</h2>
                <p>Overall {current.final_score.toFixed(1)} · {current.understanding_grade}</p>
              </div>
            ) : (
              <div className="empty">
                <Mic size={40} aria-hidden />
                <h3>No results yet</h3>
                <p>Choose a concept, record or upload your explanation, then select Analyse.</p>
              </div>
            )}
          </section>
        </main>
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
