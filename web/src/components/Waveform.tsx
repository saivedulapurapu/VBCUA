import { useEffect, useRef } from 'react'

/** Live microphone waveform drawn from an AnalyserNode (accent colour from the theme). */
export function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !analyser) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const data = new Uint8Array(analyser.fftSize)
    let frame = 0

    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      if (canvas.width !== width * dpr) {
        canvas.width = width * dpr
        canvas.height = height * dpr
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)
      analyser.getByteTimeDomainData(data)
      const styles = getComputedStyle(canvas)
      const gradient = ctx.createLinearGradient(0, 0, width, 0)
      gradient.addColorStop(0, styles.getPropertyValue('--accent').trim() || '#6366f1')
      gradient.addColorStop(1, styles.getPropertyValue('--accent-2').trim() || '#06b6d4')
      ctx.lineWidth = 2
      ctx.strokeStyle = gradient
      ctx.beginPath()
      const step = width / data.length
      for (let i = 0; i < data.length; i++) {
        const y = (data[i] / 255) * height
        if (i === 0) ctx.moveTo(0, y)
        else ctx.lineTo(i * step, y)
      }
      ctx.stroke()
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frame)
  }, [analyser])

  return <canvas ref={canvasRef} className="wave" aria-hidden />
}
