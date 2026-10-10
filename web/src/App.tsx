import { useEffect, useState } from 'react'
import { Mic, Plus } from 'lucide-react'
import { api } from './api'
import type { Concept } from './types'
import { initialConceptId } from './lib/display'
import { useTheme } from './hooks/useTheme'
import { useUrlParam } from './hooks/useUrlParam'
import { ThemeToggle } from './components/ThemeToggle'
import { ConceptPicker } from './components/ConceptPicker'
import { AddConceptDrawer } from './components/AddConceptDrawer'
import { ErrorNotice } from './components/ErrorNotice'

export default function App() {
  const { theme, cycle } = useTheme()
  const [conceptParam, setConceptParam] = useUrlParam('concept')
  const [concepts, setConcepts] = useState<Concept[]>([])
  const [loadError, setLoadError] = useState<unknown>(null)
  const [addingConcept, setAddingConcept] = useState(false)

  useEffect(() => {
    api.getConcepts().then(setConcepts).catch(setLoadError)
  }, [])

  const selectedId = initialConceptId(concepts, conceptParam)
  const selectConcept = (id: number) => setConceptParam(String(id))

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
          </section>

          <section className="panel glass" aria-label="Results">
            <div className="empty">
              <Mic size={40} aria-hidden />
              <h3>No results yet</h3>
              <p>Choose a concept, record or upload your explanation, then select Analyse.</p>
            </div>
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
