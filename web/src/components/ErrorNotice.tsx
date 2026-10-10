import { AlertTriangle } from 'lucide-react'
import { ApiError } from '../api'

/** A short message, with the technical detail available on demand. */
export function ErrorNotice({ error }: { error: unknown }) {
  const message = error instanceof ApiError ? error.message : 'Something went wrong. Please try again.'
  const detail = error instanceof ApiError ? error.detail : String(error)
  return (
    <div className="notice notice-error" role="alert">
      <AlertTriangle size={18} aria-hidden />
      <div>
        <div className="notice-title">{message}</div>
        {detail && (
          <details>
            <summary className="small muted">Technical details</summary>
            <pre>{detail}</pre>
          </details>
        )}
      </div>
    </div>
  )
}
