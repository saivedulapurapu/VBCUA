import { useId, useMemo, useRef, useState } from 'react'
import { ChevronDown, Shuffle, Sparkles } from 'lucide-react'
import type { Concept } from '../types'
import { randomConcept } from '../lib/display'

interface Props {
  concepts: Concept[]
  selectedId: number | null
  onSelect: (id: number) => void
}

/** Searchable concept list with keyboard support and a "Surprise me" button. */
export function ConceptPicker({ concepts, selectedId, onSelect }: Props) {
  const selected = concepts.find((c) => c.id === selectedId) ?? null
  const [query, setQuery] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const listId = useId()
  const inputRef = useRef<HTMLInputElement>(null)

  const matches = useMemo(() => {
    const q = (query ?? '').trim().toLowerCase()
    if (!q) return concepts
    return concepts.filter((c) => c.title.toLowerCase().includes(q) || c.keywords.some((k) => k.toLowerCase().includes(q)))
  }, [concepts, query])

  const choose = (concept: Concept) => {
    onSelect(concept.id)
    setQuery(null)
    setOpen(false)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setOpen(true)
      setActive((i) => Math.min(i + 1, matches.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && open && matches[active]) {
      e.preventDefault()
      choose(matches[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
      setQuery(null)
    }
  }

  const surprise = () => {
    const next = randomConcept(concepts, selectedId)
    if (next) choose(next)
  }

  return (
    <div className="picker">
      <div className="picker-row">
        <div style={{ position: 'relative' }}>
          <input
            ref={inputRef}
            className="field"
            role="combobox"
            aria-expanded={open}
            aria-controls={listId}
            aria-autocomplete="list"
            aria-label="Concept"
            placeholder="Search concepts…"
            value={query ?? selected?.title ?? ''}
            onChange={(e) => {
              setQuery(e.target.value)
              setOpen(true)
              setActive(0)
            }}
            onFocus={(e) => {
              setOpen(true)
              e.target.select()
            }}
            onBlur={() => setTimeout(() => {
              setOpen(false)
              setQuery(null)
            }, 120)}
            onKeyDown={onKeyDown}
          />
          <ChevronDown size={16} aria-hidden style={{ position: 'absolute', right: 12, top: 14, opacity: 0.6, pointerEvents: 'none' }} />
          {open && (
            <ul className="options glass elevated" id={listId} role="listbox">
              {matches.length === 0 && <li className="option-empty">No matching concept</li>}
              {matches.map((c, i) => (
                <li
                  key={c.id}
                  role="option"
                  className="option"
                  aria-selected={i === active}
                  onMouseDown={(e) => {
                    e.preventDefault()
                    choose(c)
                  }}
                  onMouseEnter={() => setActive(i)}
                >
                  {c.title}
                  {c.source === 'ai-generated' && <span className="option-tag">AI reference</span>}
                </li>
              ))}
            </ul>
          )}
        </div>
        <button type="button" className="btn" onClick={surprise} disabled={concepts.length < 2} title="Pick a random concept">
          <Shuffle size={16} aria-hidden /> Surprise me
        </button>
      </div>

      {selected?.source === 'ai-generated' && (
        <div className="source-badge" title="The reference definition and keywords for this topic were written by AI. Scores are less reliable than for the curated concepts.">
          <Sparkles size={14} aria-hidden /> AI-generated reference: review it below
        </div>
      )}
      {selected && (
        <details className="disclosure" open={selected.source === 'ai-generated' ? true : undefined}>
          <summary>Reference definition and keywords</summary>
          <p>{selected.description}</p>
          <div className="chips">
            {selected.keywords.map((k) => (
              <span key={k} className="chip chip-plain">{k}</span>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}
