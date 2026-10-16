import { useId } from 'react'

const SIZE = 96
const STROKE = 9
const RADIUS = (SIZE - STROKE) / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

/**
 * Circular 0–100 score. The arc is drawn at its final value and a CSS animation fills it
 * from empty, so it is always correct even when animation frames are paused (background tab).
 */
export function ScoreRing({ value, label }) {
  // useId can contain characters (e.g. ':' or '«') that break url(#…) references in SVG
  const gradientId = `ring-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const fraction = Math.max(0, Math.min(100, value ?? 0)) / 100

  return (
    <div className="ring" role="img" aria-label={`${label}: ${value === null ? 'not available' : value.toFixed(1)} out of 100`}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden>
        <defs>
          {/* CSS variables only work in style, not in SVG presentation attributes */}
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" style={{ stopColor: 'var(--accent)' }} />
            <stop offset="1" style={{ stopColor: 'var(--accent-2)' }} />
          </linearGradient>
        </defs>
        <circle cx={SIZE / 2} cy={SIZE / 2} r={RADIUS} fill="none" stroke="var(--ring-track)" strokeWidth={STROKE} />
        <circle
          key={value ?? 'none'}
          className="ring-arc"
          style={{ '--circumference': CIRCUMFERENCE } }
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke={`url(#${gradientId})`}
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
