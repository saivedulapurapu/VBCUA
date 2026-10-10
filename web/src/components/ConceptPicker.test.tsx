import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConceptPicker } from './ConceptPicker'
import { CONCEPTS } from '../test/fixtures'

describe('ConceptPicker', () => {
  it('shows the selected concept and its reference', () => {
    render(<ConceptPicker concepts={CONCEPTS} selectedId={1} onSelect={() => {}} />)
    expect(screen.getByRole('combobox')).toHaveValue('Machine Learning')
    expect(screen.getByText('ML def')).toBeInTheDocument()
  })

  it('filters by title or keyword and selects with the keyboard', async () => {
    const onSelect = vi.fn()
    render(<ConceptPicker concepts={CONCEPTS} selectedId={1} onSelect={onSelect} />)
    await userEvent.click(screen.getByRole('combobox'))
    await userEvent.keyboard('sql')
    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['Database Management System (DBMS)'])
    await userEvent.keyboard('{Enter}')
    expect(onSelect).toHaveBeenCalledWith(3)
  })

  it('surprise me picks a different concept', async () => {
    const onSelect = vi.fn()
    render(<ConceptPicker concepts={CONCEPTS} selectedId={1} onSelect={onSelect} />)
    await userEvent.click(screen.getByRole('button', { name: /surprise me/i }))
    expect(onSelect).toHaveBeenCalledOnce()
    expect(onSelect.mock.calls[0][0]).not.toBe(1)
  })

  it('renders hostile concept text as plain text', () => {
    const hostile = [{ id: 1, title: '<img src=x onerror=alert(1)>', description: '<script>alert(1)</script>', keywords: ['<b>k</b>'] }]
    const { container } = render(<ConceptPicker concepts={hostile} selectedId={1} onSelect={() => {}} />)
    expect(container.querySelector('script, img[src="x"], b')).toBeNull()
    expect(screen.getByText('<script>alert(1)</script>')).toBeInTheDocument()
  })
})
