import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { feedbackInProgress } from '../lib/display'


const POLL_MS = 3000
const MAX_WAIT_MS = 300_000

/**
 * While an attempt's AI feedback is pending, asks the backend to write it (the backend never
 * calls the AI twice for one attempt) and, if another request is already generating it, polls
 * until it is done. Stops when a different attempt is shown.
 */
export function useFeedback(current, onUpdate) {
  const [error, setError] = useState(null)
  const onUpdateRef = useRef(onUpdate)
  onUpdateRef.current = onUpdate

  const id = current?.id
  const waiting = current ? feedbackInProgress(current) : false

  useEffect(() => {
    if (id === undefined || !waiting) return
    const controller = new AbortController()
    const { signal } = controller
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
    setError(null)

    ;(async () => {
      try {
        let result = await api.requestFeedback(id, signal)
        const started = Date.now()
        while (result.feedback_status === 'generating' && Date.now() - started < MAX_WAIT_MS) {
          await sleep(POLL_MS)
          if (signal.aborted) return
          result = await api.getEvaluation(id, signal)
        }
        if (!signal.aborted) onUpdateRef.current(result)
      } catch (e) {
        if (!signal.aborted) setError(e)
      }
    })()
    return () => controller.abort()
  }, [id, waiting])

  return { error }
}
