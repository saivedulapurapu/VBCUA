import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { api } from '../api'

import { ErrorNotice } from './ErrorNotice'






/** Side panel to add a concept with a reference definition and keywords. */
export function AddConceptDrawer({ onClose, onCreated }) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [keywords, setKeywords] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const firstField = useRef(null)

  useEffect(() => {
    firstField.current?.focus()
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const keywordList = keywords.split(',').map((k) => k.trim()).filter(Boolean)
  const valid = title.trim() && description.trim() && keywordList.length > 0

  const submit = async (e) => {
    e.preventDefault()
    if (!valid) return
    setSaving(true)
    setError(null)
    try {
      onCreated(await api.createConcept(title.trim(), description.trim(), keywordList))
    } catch (err) {
      setError(err)
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      <div className="drawer-overlay" onClick={onClose} />
      <aside className="drawer glass elevated" role="dialog" aria-modal="true" aria-labelledby="add-concept-title">
        <div className="result-head">
          <h2 id="add-concept-title" className="result-title">Add a concept</h2>
          <button type="button" className="btn btn-icon" onClick={onClose} aria-label="Close">
            <X size={18} aria-hidden />
          </button>
        </div>
        <p className="muted small">Add your own topic with a reference definition and the keywords a good explanation should mention.</p>
        <form className="form" onSubmit={submit}>
          <label>
            Title
            <input ref={firstField} className="field" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. REST API" maxLength={100} />
          </label>
          <label>
            Reference definition
            <textarea className="field" value={description} onChange={(e) => setDescription(e.target.value)} />
          </label>
          <label>
            Keywords (comma separated)
            <input className="field" value={keywords} onChange={(e) => setKeywords(e.target.value)} placeholder="e.g. client, server, HTTP" />
          </label>
          {error ? <ErrorNotice error={error} /> : null}
          <button type="submit" className="btn btn-primary" disabled={!valid || saving}>
            {saving ? 'Adding…' : 'Add concept'}
          </button>
        </form>
      </aside>
    </>
  )
}
