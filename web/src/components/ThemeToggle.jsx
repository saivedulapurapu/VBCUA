import { Monitor, Moon, Sun } from 'lucide-react'


const LABELS = { system: 'System theme', light: 'Light theme', dark: 'Dark theme' }

export function ThemeToggle({ theme, onCycle }) {
  const Icon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor
  return (
    <button type="button" className="btn btn-icon" onClick={onCycle} title={`${LABELS[theme]} (click to change)`} aria-label={`${LABELS[theme]}. Click to change theme.`}>
      <Icon size={18} aria-hidden />
    </button>
  )
}
