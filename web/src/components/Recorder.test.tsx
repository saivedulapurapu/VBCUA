import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { isAcceptedAudio, Recorder } from './Recorder'

afterEach(() => vi.unstubAllGlobals())

describe('isAcceptedAudio', () => {
  it('accepts supported extensions only', () => {
    expect(isAcceptedAudio('talk.WAV')).toBe(true)
    expect(isAcceptedAudio('talk.m4a')).toBe(true)
    expect(isAcceptedAudio('notes.txt')).toBe(false)
  })
})

describe('Recorder', () => {
  it('starts in record mode with no audio', () => {
    const onChange = vi.fn()
    render(<Recorder onChange={onChange} />)
    expect(screen.getByRole('button', { name: 'Start recording' })).toBeInTheDocument()
    expect(onChange).toHaveBeenLastCalledWith(null)
  })

  it('reports an uploaded file', async () => {
    vi.stubGlobal('URL', { ...URL, createObjectURL: () => 'blob:x', revokeObjectURL: () => {} })
    const onChange = vi.fn()
    render(<Recorder onChange={onChange} />)
    await userEvent.click(screen.getByRole('button', { name: 'Upload' }))
    const file = new File(['abc'], 'talk.mp3', { type: 'audio/mpeg' })
    await userEvent.upload(screen.getByLabelText('Audio file'), file)
    expect(screen.getByText('talk.mp3')).toBeInTheDocument()
    expect(onChange).toHaveBeenLastCalledWith({ blob: file, filename: 'talk.mp3' })
  })

  it('rejects unsupported files', async () => {
    const onChange = vi.fn()
    render(<Recorder onChange={onChange} />)
    await userEvent.click(screen.getByRole('button', { name: 'Upload' }))
    await userEvent.upload(screen.getByLabelText('Audio file'), new File(['x'], 'notes.txt'), { applyAccept: false })
    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported file type')
    expect(onChange).toHaveBeenLastCalledWith(null)
  })

  it('explains a blocked microphone', async () => {
    vi.stubGlobal('MediaRecorder', class {})
    vi.stubGlobal('navigator', {
      ...navigator,
      mediaDevices: { getUserMedia: () => Promise.reject(new DOMException('denied', 'NotAllowedError')) },
    })
    render(<Recorder onChange={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'Start recording' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Microphone access was blocked'))
  })

  it('explains when recording is not supported', async () => {
    vi.stubGlobal('navigator', { ...navigator, mediaDevices: undefined })
    render(<Recorder onChange={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'Start recording' }))
    expect(screen.getByRole('alert')).toHaveTextContent('not supported')
  })
})
