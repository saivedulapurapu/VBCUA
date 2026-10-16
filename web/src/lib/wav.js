/**
 * Converts browser recordings (usually WebM/Opus) to 16 kHz mono 16-bit WAV before upload:
 * one predictable format for the backend, and a small file (~1.9 MB per minute).
 */

export const TARGET_SAMPLE_RATE = 16_000

/** Encodes mono float samples in [-1, 1] as a 16-bit PCM WAV file. */
export function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2
  const dataSize = samples.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i))
  }
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * bytesPerSample, true) // byte rate
  view.setUint16(32, bytesPerSample, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeString(36, 'data')
  view.setUint32(40, dataSize, true)
  let offset = 44
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

/** Decodes any browser-supported audio, downmixes to mono and resamples to 16 kHz. */
export async function toWav16k(input) {
  const arrayBuffer = await input.arrayBuffer()
  const decodeContext = new AudioContext()
  let decoded
  try {
    decoded = await decodeContext.decodeAudioData(arrayBuffer)
  } finally {
    void decodeContext.close()
  }
  const length = Math.ceil(decoded.duration * TARGET_SAMPLE_RATE)
  const offline = new OfflineAudioContext(1, Math.max(1, length), TARGET_SAMPLE_RATE)
  const source = offline.createBufferSource()
  source.buffer = decoded // multi-channel input is downmixed to the 1-channel destination
  source.connect(offline.destination)
  source.start()
  const rendered = await offline.startRendering()
  return encodeWav(rendered.getChannelData(0), TARGET_SAMPLE_RATE)
}
