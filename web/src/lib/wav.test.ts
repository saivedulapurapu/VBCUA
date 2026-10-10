import { describe, expect, it } from 'vitest'
import { encodeWav, TARGET_SAMPLE_RATE } from './wav'

async function bytes(blob: Blob): Promise<DataView> {
  return new DataView(await blob.arrayBuffer())
}

const text = (view: DataView, offset: number, length: number) =>
  String.fromCharCode(...Array.from({ length }, (_, i) => view.getUint8(offset + i)))

describe('encodeWav', () => {
  it('writes a 16-bit mono PCM header', async () => {
    const view = await bytes(encodeWav(new Float32Array([0, 0.5, -0.5]), TARGET_SAMPLE_RATE))
    expect(text(view, 0, 4)).toBe('RIFF')
    expect(text(view, 8, 4)).toBe('WAVE')
    expect(view.getUint16(20, true)).toBe(1) // PCM
    expect(view.getUint16(22, true)).toBe(1) // mono
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint16(34, true)).toBe(16) // bits per sample
    expect(text(view, 36, 4)).toBe('data')
    expect(view.getUint32(40, true)).toBe(6) // 3 samples x 2 bytes
    expect(view.byteLength).toBe(44 + 6)
  })

  it('encodes and clips samples', async () => {
    const view = await bytes(encodeWav(new Float32Array([1, -1, 2, -2, 0]), 16000))
    const samples = Array.from({ length: 5 }, (_, i) => view.getInt16(44 + i * 2, true))
    expect(samples).toEqual([32767, -32768, 32767, -32768, 0])
  })

  it('has the audio/wav type', () => {
    expect(encodeWav(new Float32Array(1), 16000).type).toBe('audio/wav')
  })
})
