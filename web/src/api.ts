import type { Concept, Evaluation, TrendingResponse } from './types'

/** A failed request: a short message for the user and the technical detail kept separately. */
export class ApiError extends Error {
  readonly detail: string
  readonly status?: number

  constructor(message: string, detail = '', status?: number) {
    super(message)
    this.name = 'ApiError'
    this.detail = detail
    this.status = status
  }
}

// Whisper on CPU plus scoring can take a while for long recordings
const EVALUATE_TIMEOUT_MS = 600_000
const FEEDBACK_TIMEOUT_MS = 300_000
const DEFAULT_TIMEOUT_MS = 15_000

function statusMessage(status: number): string {
  if (status === 404) return 'The requested item was not found.'
  if (status === 503) return 'AI features are not configured on the server (no Gemini API key).'
  if (status === 502) return 'The AI service could not prepare this right now. Please try again later.'
  if (status === 400 || status === 422) return 'The server could not process this request. Check the inputs and try again.'
  return 'The analysis failed on the server. Please try again.'
}

async function request(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const outer = init.signal
  outer?.addEventListener('abort', () => controller.abort(), { once: true })
  let response: Response
  try {
    response = await fetch(path, { ...init, signal: controller.signal })
  } catch (e) {
    if (outer?.aborted) throw e
    if (controller.signal.aborted) {
      throw new ApiError('The server is taking longer than expected. The result may still be saved: check History in a minute.', String(e))
    }
    throw new ApiError('Cannot reach the analysis server. Make sure the backend is running.', String(e))
  } finally {
    clearTimeout(timer)
  }
  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? body)
    } catch {
      detail = await response.text().catch(() => '')
    }
    throw new ApiError(statusMessage(response.status), `HTTP ${response.status}: ${detail}`, response.status)
  }
  return response
}

const json = async <T,>(res: Promise<Response>): Promise<T> => (await res).json() as Promise<T>

export const api = {
  getConcepts: () => json<Concept[]>(request('/api/concepts')),

  createConcept: (title: string, description: string, keywords: string[]) =>
    json<Concept>(request('/api/concepts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, description, keywords }),
    })),

  evaluate: (conceptId: number, audio: Blob, filename: string, signal?: AbortSignal) => {
    const form = new FormData()
    form.append('concept_id', String(conceptId))
    form.append('file', audio, filename)
    return json<Evaluation>(request('/api/evaluate', { method: 'POST', body: form, signal }, EVALUATE_TIMEOUT_MS))
  },

  /** Today's topics; the backend caches them for the day. */
  getTrending: () => json<TrendingResponse>(request('/api/discover/trending', {}, 60_000)),

  /** Creates (or returns the existing) concept for any topic, with an AI-written reference. */
  generateConcept: (topic: string) =>
    json<Concept>(request('/api/concepts/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic }),
    }, 60_000)),

  getEvaluations: () => json<Evaluation[]>(request('/api/evaluations')),

  getEvaluation: (id: number, signal?: AbortSignal) => json<Evaluation>(request(`/api/evaluations/${id}`, { signal })),

  /** Safe to call more than once: the backend never calls the AI twice for an attempt. */
  requestFeedback: (id: number, signal?: AbortSignal) =>
    json<Evaluation>(request(`/api/evaluations/${id}/feedback`, { method: 'POST', signal }, FEEDBACK_TIMEOUT_MS)),

  pdfUrl: (id: number) => `/api/evaluations/${id}/pdf`,
}
