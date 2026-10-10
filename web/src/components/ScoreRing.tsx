import { useEffect, useState } from 'react'

const SIZE = 96
const STROKE = 9
const RADIUS = (SIZE - STROKE) / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

/** Circular 0–100 score that animates from empty when shown. */
export function ScoreRing({ value, label }: { value: number | null; label: string }) {
  const [shown, setShown] = useState(0)
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(value ?? 0))
    return () => cancelAnimationFrame(id)
  }, [value])
  const fraction = Math.max(0, Math.min(100, shown)) / 100

  return (
    <div className="ring" role="img" aria-label={`${label}: ${value === null ? 'not available' : value.toFixed(1)} out of 100`}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden>
        <defs>
          <linearGradient id="ring-gradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--accent)" />
            <stop offset="1" stopColor="var(--accent-2)" />
          </linearGradient>
        </defs>
        <circle cx={SIZE / 2} cy={SIZE / 2} r={RADIUS} fill="none" stroke="var(--ring-track)" strokeWidth={STROKE} />
        <circle
          className="ring-arc"
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke="url(#ring-gradient)"
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={CIRCUMFERENCE * (1 - fraction)}
        />
      </svg>
      <div className="ring-value" aria-hidden>{value === null ? 'n/a' : Math.round(value)}</div>
    </div>
  )
}
