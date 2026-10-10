import { useCallback, useEffect, useRef, useState } from 'react'
import { toWav16k } from '../lib/wav'

export type RecorderState = 'idle' | 'recording' | 'processing' | 'recorded' | 'error'

export interface Recording {
  blob: Blob
  filename: string
  url: string
  seconds: number
}

/**
 * Microphone recording with a live level analyser for the waveform.
 * The result is converted to 16 kHz mono WAV (falls back to the original if conversion fails).
 */
export function useRecorder() {
  const [state, setState] = useState<RecorderState>('idle')
  const [seconds, setSeconds] = useState(0)
  const [recording, setRecording] = useState<Recording | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const audioContext = useRef<AudioContext | null>(null)
  const chunks = useRef<Blob[]>([])
  const startedAt = useRef(0)
  const timer = useRef<number | null>(null)

  const cleanupStream = useCallback(() => {
    if (timer.current !== null) window.clearInterval(timer.current)
    timer.current = null
    stream.current?.getTracks().forEach((t) => t.stop())
    stream.current = null
    void audioContext.current?.close()
    audioContext.current = null
    setAnalyser(null)
  }, [])

  const clear = useCallback(() => {
    setRecording((r) => {
      if (r) URL.revokeObjectURL(r.url)
      return null
    })
    setSeconds(0)
    setError(null)
    setState('idle')
  }, [])

  const start = useCallback(async () => {
    clear()
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('Recording is not supported in this browser. Use Upload instead.')
      setState('error')
      return
    }
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
      stream.current = media
      const ctx = new AudioContext()
      audioContext.current = ctx
      const node = ctx.createAnalyser()
      node.fftSize = 1024
      ctx.createMediaStreamSource(media).connect(node)
      setAnalyser(node)

      chunks.current = []
      const recorder = new MediaRecorder(media)
      mediaRecorder.current = recorder
      recorder.ondataavailable = (e) => e.data.size > 0 && chunks.current.push(e.data)
      recorder.onstop = async () => {
        const elapsed = (performance.now() - startedAt.current) / 1000
        cleanupStream()
        setState('processing')
        const original = new Blob(chunks.current, { type: recorder.mimeType || 'audio/webm' })
        let blob = original
        let filename = 'recording.webm'
        try {
          blob = await toWav16k(original)
          filename = 'recording.wav'
        } catch {
          /* keep the original format; the backend decodes it with FFmpeg */
        }
        setRecording({ blob, filename, url: URL.createObjectURL(blob), seconds: elapsed })
        setState('recorded')
      }
      startedAt.current = performance.now()
      recorder.start()
      setState('recording')
      timer.current = window.setInterval(() => setSeconds((performance.now() - startedAt.current) / 1000), 250)
    } catch (e) {
      cleanupStream()
      const denied = e instanceof DOMException && (e.name === 'NotAllowedError' || e.name === 'SecurityError')
      setError(denied
        ? 'Microphone access was blocked. Allow it in the browser address bar, or use Upload instead.'
        : 'No microphone could be started. Check that one is connected, or use Upload instead.')
      setState('error')
    }
  }, [clear, cleanupStream])

  const stop = useCallback(() => {
    if (mediaRecorder.current?.state === 'recording') mediaRecorder.current.stop()
  }, [])

  useEffect(() => () => {
    if (mediaRecorder.current?.state === 'recording') mediaRecorder.current.stop()
    cleanupStream()
  }, [cleanupStream])

  return { state, seconds, recording, error, analyser, start, stop, clear }
}
