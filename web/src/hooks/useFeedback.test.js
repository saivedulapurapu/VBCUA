import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { api, ApiError } from '../api'
import { attempt } from '../test/fixtures'
import { useFeedback } from './useFeedback'

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

const pending = attempt({ feedback_source: 'rule-based', feedback_status: 'pending', feedback: 'rule text' })
const done = attempt({ feedback_source: 'llm', feedback_status: 'done', feedback: 'AI text' })

describe('useFeedback', () => {
  it('requests AI feedback once for a pending attempt', async () => {
    const request = vi.spyOn(api, 'requestFeedback').mockResolvedValue(done)
    const onUpdate = vi.fn()
    renderHook(() => useFeedback(pending, onUpdate))
    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(done))
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('does nothing when feedback is already done', () => {
    const request = vi.spyOn(api, 'requestFeedback')
    renderHook(() => useFeedback(done, vi.fn()))
    expect(request).not.toHaveBeenCalled()
  })

  it('polls while another request is generating', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    vi.spyOn(api, 'requestFeedback').mockResolvedValue(attempt({ feedback_source: 'rule-based', feedback_status: 'generating' }))
    const get = vi.spyOn(api, 'getEvaluation').mockResolvedValue(done)
    const onUpdate = vi.fn()
    renderHook(() => useFeedback(pending, onUpdate))
    await vi.advanceTimersByTimeAsync(3100)
    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(done))
    expect(get).toHaveBeenCalledTimes(1)
  })

  it('reports errors', async () => {
    vi.spyOn(api, 'requestFeedback').mockRejectedValue(new ApiError('Cannot reach the analysis server.'))
    const { result } = renderHook(() => useFeedback(pending, vi.fn()))
    await waitFor(() => expect(result.current.error).toBeInstanceOf(ApiError))
  })

  it('stops when a different attempt is shown', async () => {
    let resolve = () => {}
    vi.spyOn(api, 'requestFeedback').mockImplementation(() => new Promise((r) => { resolve = r }))
    const onUpdate = vi.fn()
    const { rerender } = renderHook(({ e }) => useFeedback(e, onUpdate), { initialProps: { e: pending } })
    rerender({ e: done })
    resolve(done)
    await new Promise((r) => setTimeout(r, 20))
    expect(onUpdate).not.toHaveBeenCalled()
  })
})
