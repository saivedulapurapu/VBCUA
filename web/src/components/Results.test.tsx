import { describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Results } from './Results'
import { History } from './History'
import { attempt } from '../test/fixtures'

const renderResults = (overrides = {}, onRetry = () => {}) =>
  render(<Results evaluation={attempt(overrides)} feedbackError={null} onRetryFeedback={onRetry} />)

describe('Results', () => {
  it('shows the evaluated concept, grade and three scores', () => {
    renderResults()
    expect(screen.getByRole('heading', { name: 'Cloud Computing' })).toBeInTheDocument()
    expect(screen.getByText('Moderate Understanding')).toHaveClass('badge-moderate')
    expect(screen.getByRole('img', { name: 'Overall: 76.9 out of 100' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Understanding: 67.0 out of 100' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Fluency: 100.0 out of 100' })).toBeInTheDocument()
    expect(screen.getByText('Pauses 26% of speaking time')).toBeInTheDocument()
  })

  it('switches tabs with clicks and arrow keys', async () => {
    renderResults()
    expect(screen.getByText('Good work.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('tab', { name: 'Keywords' }))
    expect(screen.getByText('1 of 2 key terms mentioned (whole-word matches).')).toBeInTheDocument()
    expect(screen.getByText('✓ servers')).toHaveClass('chip-found')
    expect(screen.getByText('✗ storage')).toHaveClass('chip-missing')
    await userEvent.keyboard('{ArrowRight}')
    expect(screen.getByRole('tab', { name: 'Transcript' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByText('We rent servers.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('tab', { name: 'Details' }))
    const table = screen.getByRole('table')
    expect(within(table).getByText('Cosine similarity')).toBeInTheDocument()
  })

  it('shows "Writing feedback…" while AI feedback is pending', () => {
    renderResults({ feedback_source: 'rule-based', feedback_status: 'pending' })
    expect(screen.getByText('Writing feedback…')).toBeInTheDocument()
    expect(screen.queryByText(/rule-based generator/)).toBeNull()
  })

  it('offers a retry after AI feedback failed', async () => {
    const onRetry = vi.fn()
    renderResults({ feedback_source: 'rule-based', feedback_status: 'failed', feedback: 'Rule text' }, onRetry)
    expect(screen.getByText(/could not be generated/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /try ai feedback again/i }))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('shows the microphone warning', () => {
    renderResults({ recording_warning: 'Recording is very quiet — check your microphone.' })
    expect(screen.getByText(/check your microphone/)).toBeInTheDocument()
  })

  it('never renders markup from transcript or feedback', async () => {
    const { container } = renderResults({
      transcription: '<img src=x onerror=alert(1)>',
      feedback: '<script>alert(1)</script>\n\n<img src=x onerror=alert(2)>\n\n**bold**',
    })
    expect(container.querySelector('script, img[src="x"]')).toBeNull()
    expect(container.querySelector('strong')?.textContent).toBe('bold')
    await userEvent.click(screen.getByRole('tab', { name: 'Transcript' }))
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeInTheDocument()
    expect(container.querySelector('img[src="x"]')).toBeNull()
  })

  it('links the PDF report for the attempt', () => {
    renderResults()
    expect(screen.getByRole('link', { name: /download pdf/i })).toHaveAttribute('href', '/api/evaluations/7/pdf')
  })
})

describe('History', () => {
  it('lists newest first and opens an attempt', async () => {
    const onOpen = vi.fn()
    const older = attempt({ id: 1, final_score: 51.5 })
    const newer = attempt({ id: 2, final_score: 82.2, understanding_grade: 'Strong Understanding' })
    render(<History items={[older, newer]} currentId={1} onOpen={onOpen} />)
    const buttons = screen.getAllByRole('button')
    expect(buttons.map((b) => b.querySelector('.history-score')?.textContent)).toEqual(['82.2', '51.5'])
    expect(buttons[1]).toHaveAttribute('aria-current', 'true')
    await userEvent.click(buttons[0])
    expect(onOpen).toHaveBeenCalledWith(newer)
  })

  it('has an empty state', () => {
    render(<History items={[]} currentId={null} onOpen={() => {}} />)
    expect(screen.getByText(/No attempts yet/)).toBeInTheDocument()
  })
})

describe('Practise again', () => {
  it('is offered only when the attempt concept differs from the selection', async () => {
    const onPractise = vi.fn()
    const { rerender } = render(<Results evaluation={attempt()} feedbackError={null} onRetryFeedback={() => {}} canPractise onPractise={onPractise} />)
    await userEvent.click(screen.getByRole('button', { name: /practise this concept again/i }))
    expect(onPractise).toHaveBeenCalledOnce()
    rerender(<Results evaluation={attempt()} feedbackError={null} onRetryFeedback={() => {}} canPractise={false} onPractise={onPractise} />)
    expect(screen.queryByRole('button', { name: /practise this concept again/i })).toBeNull()
  })
})
