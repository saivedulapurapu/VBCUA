import { useEffect, useState } from 'react'

/** Seconds since `running` became true (0 while not running). */
export function useElapsed(running) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    if (!running) {
      setSeconds(0)
      return
    }
    const start = performance.now()
    const id = window.setInterval(() => setSeconds((performance.now() - start) / 1000), 500)
    return () => window.clearInterval(id)
  }, [running])
  return seconds
}
