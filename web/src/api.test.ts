import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError } from './api'

afterEach(() => vi.unstubAllGlobals())

function respond(status: number, body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })),
  )
}

describe('api errors', () => {
  it('network failure has a friendly message', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }))
    const error = await api.getConcepts().catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error.message).toContain('backend is running')
    expect(error.detail).toContain('Failed to fetch')
  })

  it('server errors keep the raw detail out of the message', async () => {
    respond(500, { detail: 'Transcription failed: [WinError 2]' })
    const error = await api.getEvaluations().catch((e) => e)
    expect(error.message).not.toContain('WinError')
    expect(error.detail).toContain('WinError')
    expect(error.status).toBe(500)
  })

  it('returns parsed JSON on success', async () => {
    respond(200, [{ id: 1 }])
    expect(await api.getConcepts()).toEqual([{ id: 1 }])
  })

  it('evaluate posts the concept and audio as a form', async () => {
    const fetchMock = vi.fn(async () => new Response('{"id": 4}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await api.evaluate(3, new Blob(['x'], { type: 'audio/wav' }), 'rec.wav')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/evaluate')
    const form = init.body as FormData
    expect(form.get('concept_id')).toBe('3')
    expect((form.get('file') as File).name).toBe('rec.wav')
  })
})
