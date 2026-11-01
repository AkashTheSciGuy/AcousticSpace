/** Validation-only service. No deepfake classification is performed here. */
export type AudioValidationResult = {
  status: 'valid'
  filename: string
  format: 'wav' | 'mp3'
  size_bytes: number
  message: string
}

const MOCK_MODE = import.meta.env.VITE_AUDIO_API_MODE !== 'live'

export const validationMode: 'mock' | 'live' = MOCK_MODE ? 'mock' : 'live'

export class AudioApiError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AudioApiError'
  }
}

function isValidationResult(value: unknown): value is AudioValidationResult {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  return (
    item.status === 'valid' &&
    typeof item.filename === 'string' &&
    (item.format === 'wav' || item.format === 'mp3') &&
    typeof item.size_bytes === 'number' &&
    Number.isFinite(item.size_bytes) &&
    typeof item.message === 'string'
  )
}

function mockValidate(file: File, signal: AbortSignal): Promise<AudioValidationResult> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Validation cancelled', 'AbortError'))
      return
    }

    const extension = file.name.split('.').pop()?.toLowerCase()
    const onAbort = () => {
      window.clearTimeout(timer)
      reject(new DOMException('Validation cancelled', 'AbortError'))
    }
    const timer = window.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve({
        status: 'valid',
        filename: file.name,
        format: extension === 'mp3' ? 'mp3' : 'wav',
        size_bytes: file.size,
        message: 'Mock validation complete. No server analysis was performed.',
      })
    }, 650)

    signal.addEventListener('abort', onAbort, { once: true })
  })
}

export async function validateAudio(file: File, signal: AbortSignal): Promise<AudioValidationResult> {
  if (MOCK_MODE) return mockValidate(file, signal)

  const body = new FormData()
  body.append('file', file, file.name)

  let response: Response
  try {
    // Vite's development proxy forwards /api to the local FastAPI service.
    response = await fetch('/api/audio/validate', { method: 'POST', body, signal })
  } catch (cause) {
    if (signal.aborted) throw cause
    throw new AudioApiError('Could not reach the backend. Check that FastAPI is running.')
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new AudioApiError('The API returned invalid JSON.')
  }

  if (!response.ok) {
    const detail = payload && typeof payload === 'object' && 'detail' in payload
      ? (payload as { detail: unknown }).detail
      : null
    throw new AudioApiError(typeof detail === 'string' ? detail : `Upload failed (HTTP ${response.status}).`)
  }

  if (!isValidationResult(payload)) {
    throw new AudioApiError('The API returned an unexpected validation response.')
  }
  return payload
}
