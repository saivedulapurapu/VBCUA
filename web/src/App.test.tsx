import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { api, ApiError } from './api'
import { attempt, CONCEPTS } from './test/fixtures'

beforeEach(() => {
  window.history.replaceState(null, '', '/')
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} }))
  vi.spyOn(api, 'getConcepts').mockResolvedValue(CONCEPTS)
  vi.spyOn(api, 'getEvaluations').mockResolvedValue([])
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

async function uploadAudio() {
  await userEvent.click(screen.getByRole('button', { name: 'Upload' }))
  await userEvent.upload(screen.getByLabelText('Audio file'), new File(['abc'], 'talk.wav', { type: 'audio/wav' }))
}

describe('App', () => {
  it('restores the concept from the URL and names it on the Analyse button', async () => {
    window.history.replaceState(null, '', '/?concept=3')
    render(<App />)
    const button = await screen.findByRole('button', { name: 'Analyse explanation of Database Management System (DBMS)' })
    expect(button).toBeDisabled() // no audio yet
  })

  it('analyses, shows scores first, then the AI feedback', async () => {
    const scored = attempt({ id: 11, concept_id: 3, concept: CONCEPTS[2], feedback_source: 'rule-based', feedback_status: 'pending', feedback: 'rule text' })
    const withFeedback = { ...scored, feedback_source: 'llm' as const, feedback_status: 'done' as const, feedback: 'AI says hello' }
    const evaluate = vi.spyOn(api, 'evaluate').mockResolvedValue(scored)
    let release: () => void = () => {}
    const request = vi.spyOn(api, 'requestFeedback').mockImplementation(
      () => new Promise((r) => { release = () => r(withFeedback) }),
    )
    window.history.replaceState(null, '', '/?concept=3')
    render(<App />)
    await uploadAudio()
    await userEvent.click(await screen.findByRole('button', { name: /Analyse explanation of Database/ }))

    expect(evaluate).toHaveBeenCalledWith(3, expect.any(File), 'talk.wav')
    expect(await screen.findByText('Writing feedback…')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Overall/ })).toBeInTheDocument()
    expect(window.location.search).toContain('attempt=11')

    release()
    expect(await screen.findByText('AI says hello')).toBeInTheDocument()
    expect(request).toHaveBeenCalledWith(11, expect.any(AbortSignal))
    // The new attempt is in History
    expect(screen.getAllByText('Database Management System (DBMS)').length).toBeGreaterThan(0)
  })

  it('reopens the attempt in the URL after a reload', async () => {
    window.history.replaceState(null, '', '/?attempt=7')
    vi.spyOn(api, 'getEvaluation').mockResolvedValue(attempt())
    render(<App />)
    expect(await screen.findByRole('heading', { name: 'Cloud Computing' })).toBeInTheDocument()
  })

  it('opens an attempt from History without reloading', async () => {
    vi.spyOn(api, 'getEvaluations').mockResolvedValue([attempt({ id: 4, final_score: 61.2 })])
    render(<App />)
    await userEvent.click(await screen.findByText('61.2'))
    expect(screen.getByRole('heading', { name: 'Cloud Computing' })).toBeInTheDocument()
    expect(window.location.search).toContain('attempt=4')
  })

  it('shows a friendly error when analysis fails', async () => {
    vi.spyOn(api, 'evaluate').mockRejectedValue(new ApiError('The analysis failed on the server. Please try again.', 'HTTP 500: boom'))
    render(<App />)
    await uploadAudio()
    await userEvent.click(await screen.findByRole('button', { name: /Analyse explanation of/ }))
    expect(await screen.findByText('The analysis failed on the server. Please try again.')).toBeInTheDocument()
    expect(screen.queryByText('No results yet')).toBeInTheDocument()
  })

  it('shows a friendly error when the backend is down', async () => {
    vi.spyOn(api, 'getConcepts').mockRejectedValue(new ApiError('Cannot reach the analysis server. Make sure the backend is running.'))
    render(<App />)
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Cannot reach the analysis server'))
  })

  it('cycles the theme and remembers it', async () => {
    render(<App />)
    const toggle = screen.getByRole('button', { name: /theme/i })
    await userEvent.click(toggle)
    expect(document.documentElement.dataset.theme).toBe('light')
    await userEvent.click(toggle)
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem('vbcua-theme')).toBe('dark')
    await userEvent.click(toggle)
    expect(document.documentElement.dataset.theme).toBeUndefined()
  })
})

describe('App practise again', () => {
  it('switches the picker to the opened attempt concept', async () => {
    window.history.replaceState(null, '', '/?concept=1&attempt=7')
    vi.spyOn(api, 'getEvaluation').mockResolvedValue(attempt()) // Cloud Computing (id 2)
    render(<App />)
    await userEvent.click(await screen.findByRole('button', { name: /practise this concept again/i }))
    expect(screen.getByRole('button', { name: 'Analyse explanation of Cloud Computing' })).toBeInTheDocument()
    expect(window.location.search).toContain('concept=2')
  })
})
