import { useEffect, useRef, useState } from 'react'
import { FileAudio, Mic, RotateCcw, Square, Upload } from 'lucide-react'
import { useRecorder } from '../hooks/useRecorder'
import { formatClock } from '../lib/display'
import { Waveform } from './Waveform'






const ACCEPTED = ['.wav', '.mp3', '.m4a', '.webm', '.ogg']

export function isAcceptedAudio(name) {
  const lower = name.toLowerCase()
  return ACCEPTED.some((ext) => lower.endsWith(ext))
}

/** Step 2: record in the browser (live waveform, timer) or upload / drop a file. */
export function Recorder({ onChange, disabled = false }) {
  const [mode, setMode] = useState('record')
  const recorder = useRecorder()
  const [file, setFile] = useState(null)
  const [fileError, setFileError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  // Report the active audio (from the current mode) to the parent
  useEffect(() => {
    if (mode === 'record') onChange(recorder.recording ? { blob: recorder.recording.blob, filename: recorder.recording.filename } : null)
    else onChange(file ? { blob: file.file, filename: file.file.name } : null)
  }, [mode, recorder.recording, file, onChange])

  useEffect(() => () => {
    if (file) URL.revokeObjectURL(file.url)
  }, [file])

  const pickFile = (picked) => {
    if (!picked) return
    if (!isAcceptedAudio(picked.name)) {
      setFileError(`Unsupported file type. Use ${ACCEPTED.join(', ')}.`)
      return
    }
    setFileError(null)
    setFile({ file: picked, url: URL.createObjectURL(picked) })
  }

  return (
    <div>
      <div className="segmented" role="group" aria-label="Input method">
        <button type="button" aria-pressed={mode === 'record'} onClick={() => setMode('record')} disabled={disabled}>
          Record
        </button>
        <button type="button" aria-pressed={mode === 'upload'} onClick={() => setMode('upload')} disabled={disabled}>
          Upload
        </button>
      </div>

      {mode === 'record' ? (
        <div className="recorder">
          {recorder.state === 'recording' ? (
            <>
              <button type="button" className="mic recording" onClick={recorder.stop} aria-label="Stop recording">
                <Square size={26} aria-hidden />
              </button>
              <div className="timer" aria-live="polite">{formatClock(recorder.seconds)}</div>
              <Waveform analyser={recorder.analyser} />
              <span className="small muted">Recording… click to stop</span>
            </>
          ) : recorder.state === 'processing' ? (
            <span className="small muted">Preparing audio…</span>
          ) : recorder.recording ? (
            <>
              <audio className="audio-preview" controls src={recorder.recording.url} />
              <div className="small muted">{formatClock(recorder.recording.seconds)} recorded</div>
              <button type="button" className="btn" onClick={() => void recorder.start()} disabled={disabled}>
                <RotateCcw size={16} aria-hidden /> Record again
              </button>
            </>
          ) : (
            <>
              <button type="button" className="mic" onClick={() => void recorder.start()} aria-label="Start recording" disabled={disabled}>
                <Mic size={30} aria-hidden />
              </button>
              <span className="small muted">Click the microphone to start</span>
            </>
          )}
          {recorder.error && <div className="notice notice-warn" role="alert">{recorder.error}</div>}
        </div>
      ) : (
        <div>
          <div
            className={`dropzone${dragging ? ' dragging' : ''}`}
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              pickFile(e.dataTransfer.files[0])
            }}
          >
            {file ? <FileAudio size={28} aria-hidden /> : <Upload size={28} aria-hidden />}
            {file ? <span className="file-name">{file.file.name}</span> : <span>Drop an audio file here, or click to choose</span>}
            <span className="small muted">{ACCEPTED.join(', ')}</span>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(',')}
              className="sr-only"
              aria-label="Audio file"
              onChange={(e) => pickFile(e.target.files?.[0])}
            />
          </div>
          {file && <audio className="audio-preview" style={{ marginTop: 12 }} controls src={file.url} />}
          {fileError && <div className="notice notice-warn" role="alert">{fileError}</div>}
        </div>
      )}
      <p className="small muted" style={{ marginBottom: 0 }}>Aim for 60–120 seconds, in a quiet room.</p>
    </div>
  )
}
