import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { api, ApiError } from '../api'
import type { TrendingResponse } from '../types'
import { CONCEPTS } from '../test/fixtures'
import { Discover } from './Discover'
import { ConceptPicker } from './ConceptPicker'

afterEach(() => vi.restoreAllMocks())

const LIVE: TrendingResponse = {
  date: '2026-10-10',
  origin: 'google-search',
  topics: [
    { title: 'Nobel Prize in Physics', category: 'Science', why: 'Announced this week.' },
    { title: 'Cricket World Cup final', category: 'Sports', why: 'Played today.' },
  ],
  sources: [{ title: 'Example News', uri: 'https://news.example/a' }],
  search_entry_point: '<div class="chips">Google suggestions</div>',
}
const CURATED: TrendingResponse = { ...LIVE, origin: 'curated', sources: [], search_entry_point: null }
const GENERATED = { id: 9, title: 'Nobel Prize in Physics', description: 'AI reference', keywords: ['physics'], source: 'ai-generated' as const }

describe('Discover', () => {
  it('labels live topics as trending and shows Google suggestions in a sandbox', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue(LIVE)
    const { container } = render(<Discover concepts={CONCEPTS} onConcept={() => {}} />)
    expect(await screen.findByText('Trending today')).toBeInTheDocument()
    expect(screen.getByText('Live from Google Search')).toBeInTheDocument()
    const frame = screen.getByTitle('Google Search suggestions')
    expect(frame).toHaveAttribute('sandbox', 'allow-popups allow-popups-to-escape-sandbox')
    expect(frame.getAttribute('srcdoc')).toContain('Google suggestions')
    expect(container.querySelector('.chips.topic-chips .chips')).toBeNull() // Google HTML is not injected into the page
    expect(screen.getByRole('link', { name: /Example News/ })).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('never calls the curated fallback "trending"', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue(CURATED)
    render(<Discover concepts={CONCEPTS} onConcept={() => {}} />)
    expect(await screen.findByText('Suggested topics')).toBeInTheDocument()
    expect(screen.queryByText('Trending today')).toBeNull()
    expect(screen.queryByTitle('Google Search suggestions')).toBeNull()
  })

  it('prepares a reference when a topic is chosen', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue(LIVE)
    const generate = vi.spyOn(api, 'generateConcept').mockResolvedValue(GENERATED)
    const onConcept = vi.fn()
    render(<Discover concepts={CONCEPTS} onConcept={onConcept} />)
    await userEvent.click(await screen.findByRole('button', { name: /Nobel Prize in Physics/ }))
    expect(generate).toHaveBeenCalledWith('Nobel Prize in Physics')
    await waitFor(() => expect(onConcept).toHaveBeenCalledWith(GENERATED))
  })

  it('uses an existing concept without calling the AI', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue(CURATED)
    const generate = vi.spyOn(api, 'generateConcept')
    const onConcept = vi.fn()
    render(<Discover concepts={CONCEPTS} onConcept={onConcept} />)
    await userEvent.type(screen.getByLabelText('Any topic'), 'cloud computing{Enter}')
    expect(generate).not.toHaveBeenCalled()
    expect(onConcept).toHaveBeenCalledWith(CONCEPTS[1])
  })

  it('accepts any typed topic and shows progress', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue(CURATED)
    let resolve: (c: typeof GENERATED) => void = () => {}
    vi.spyOn(api, 'generateConcept').mockImplementation(() => new Promise((r) => { resolve = r }))
    const onConcept = vi.fn()
    render(<Discover concepts={CONCEPTS} onConcept={onConcept} />)
    await userEvent.type(screen.getByLabelText('Any topic'), 'Black holes{Enter}')
    expect(screen.getByText('Preparing a reference for “Black holes”…')).toBeInTheDocument()
    resolve({ ...GENERATED, title: 'Black holes' })
    await waitFor(() => expect(onConcept).toHaveBeenCalled())
  })

  it('shows a friendly error when the AI cannot prepare a reference', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue(CURATED)
    vi.spyOn(api, 'generateConcept').mockRejectedValue(new ApiError('The AI service could not prepare this right now. Please try again later.', 'HTTP 502: 429 quota'))
    render(<Discover concepts={CONCEPTS} onConcept={() => {}} />)
    await userEvent.type(screen.getByLabelText('Any topic'), 'Black holes{Enter}')
    expect(await screen.findByText('The AI service could not prepare this right now. Please try again later.')).toBeInTheDocument()
  })

  it('still offers "any topic" when trending topics cannot load', async () => {
    vi.spyOn(api, 'getTrending').mockRejectedValue(new ApiError('down'))
    render(<Discover concepts={CONCEPTS} onConcept={() => {}} />)
    await waitFor(() => expect(screen.queryByText(/Loading topics/)).toBeNull())
    expect(screen.getByLabelText('Any topic')).toBeInTheDocument()
  })

  it('renders topic text from the server as plain text', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue({ ...CURATED, topics: [{ title: '<img src=x onerror=alert(1)>', category: 'News', why: '' }, ...CURATED.topics] })
    const { container } = render(<Discover concepts={CONCEPTS} onConcept={() => {}} />)
    expect(await screen.findByText('<img src=x onerror=alert(1)>')).toBeInTheDocument()
    expect(container.querySelector('img[src="x"]')).toBeNull()
  })
})

describe('AI-generated concepts in the picker', () => {
  it('are flagged and show their reference', () => {
    const concepts = [...CONCEPTS, { ...GENERATED }]
    render(<ConceptPicker concepts={concepts} selectedId={9} onSelect={() => {}} />)
    expect(screen.getByText(/AI-generated reference/)).toBeInTheDocument()
    expect(screen.getByText('AI reference')).toBeVisible()
  })
})
