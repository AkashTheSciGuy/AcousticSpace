import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './RirAnalyzer.css'

type Metrics = {
  onset_seconds: number
  rt60_t20_seconds: number | null
  clarity_c50_db: number | null
  direct_to_reverberant_ratio_db: number | null
  decay_fit_r_squared: number | null
  sample_rate_hz: number
  analysis_duration_seconds: number
}

type Result = {
  status: 'analyzed'
  filename: string
  message: string
  input_type: 'room_impulse_response'
  metrics: Metrics
  limitations: string[]
}

const MAX_BYTES = 20 * 1024 * 1024

function isResult(value: unknown): value is Result {
  if (!value || typeof value !== 'object') return false
  const data = value as Record<string, unknown>
  if (data.status !== 'analyzed' || data.input_type !== 'room_impulse_response' ||
      typeof data.filename !== 'string' || typeof data.message !== 'string' ||
      !Array.isArray(data.limitations) || !data.limitations.every(x => typeof x === 'string') ||
      !data.metrics || typeof data.metrics !== 'object') return false
  const m = data.metrics as Record<string, unknown>
  const required = ['onset_seconds', 'sample_rate_hz', 'analysis_duration_seconds']
  const optional = ['rt60_t20_seconds', 'clarity_c50_db', 'direct_to_reverberant_ratio_db', 'decay_fit_r_squared']
  return required.every(k => typeof m[k] === 'number' && Number.isFinite(m[k])) &&
    optional.every(k => m[k] === null || (typeof m[k] === 'number' && Number.isFinite(m[k])))
}

function numberOrUnavailable(value: number | null, unit = ''): string {
  return value === null ? 'Not estimable' : `${value.toFixed(unit === ' s' ? 3 : 2)}${unit}`
}

export default function RirAnalyzer() {
  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  function chooseFile(event: ChangeEvent<HTMLInputElement>): void {
    abortRef.current?.abort()
    setLoading(false)
    setResult(null)
    setError(null)
    const selected = event.target.files?.[0]
    event.target.value = ''
    if (!selected) return
    if (!selected.name.toLowerCase().endsWith('.wav')) {
      setFile(null)
      setError('Choose a WAV impulse-response recording.')
    } else if (selected.size === 0 || selected.size > MAX_BYTES) {
      setFile(null)
      setError('RIR must be nonempty and at most 20 MiB.')
    } else {
      setFile(selected)
    }
  }

  async function analyze(): Promise<void> {
    if (!file || loading) return
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true)
    setError(null)
    setResult(null)
    const timeout = window.setTimeout(() => controller.abort(), 30000)
    try {
      const form = new FormData()
      form.append('file', file, file.name)
      const response = await fetch('/api/rir/analyze', { method: 'POST', body: form, signal: controller.signal })
      const data: unknown = await response.json()
      if (!response.ok) {
        const detail = data && typeof data === 'object' && 'detail' in data ? (data as { detail: unknown }).detail : null
        throw new Error(typeof detail === 'string' ? detail : `RIR analysis failed (HTTP ${response.status}).`)
      }
      if (!isResult(data)) throw new Error('The server returned an invalid RIR response.')
      if (abortRef.current === controller) setResult(data)
    } catch (cause) {
      if (abortRef.current === controller) {
        setError(controller.signal.aborted ? 'RIR request cancelled or timed out.' :
          cause instanceof Error ? cause.message : 'Could not analyze RIR.')
      }
    } finally {
      window.clearTimeout(timeout)
      if (abortRef.current === controller) {
        setLoading(false)
        abortRef.current = null
      }
    }
  }

  return (
    <section className="rir-card" aria-labelledby="rir-title">
      <span className="section-label">STEP 02 · ROOM ACOUSTICS</span>
      <h2 id="rir-title">Analyze a room impulse response</h2>
      <p className="rir-description">
        Upload a measured or simulated <strong>RIR WAV</strong>, not normal speech.
        The backend estimates reverberation time and clarity from the impulse response.
      </p>
      <div className="rir-controls">
        <input ref={inputRef} type="file" accept=".wav,audio/wav" onChange={chooseFile} aria-label="Choose impulse-response WAV" />
        <button type="button" onClick={() => { void analyze() }} disabled={!file || loading}>
          {loading ? 'Analyzing…' : 'Analyze room acoustics'}
        </button>
      </div>
      {file && <p className="rir-filename">Selected: {file.name}</p>}
      {error && <p role="alert" className="rir-error">{error}</p>}
      {result && (
        <div className="rir-results" role="status">
          <h3>RIR measurements</h3>
          <div className="rir-metrics">
            <div><span>RT60 (T20 estimate)</span><strong>{numberOrUnavailable(result.metrics.rt60_t20_seconds, ' s')}</strong></div>
            <div><span>Clarity (C50)</span><strong>{numberOrUnavailable(result.metrics.clarity_c50_db, ' dB')}</strong></div>
            <div><span>Approx. direct/reverb ratio</span><strong>{numberOrUnavailable(result.metrics.direct_to_reverberant_ratio_db, ' dB')}</strong></div>
            <div><span>Decay fit R²</span><strong>{numberOrUnavailable(result.metrics.decay_fit_r_squared)}</strong></div>
            <div><span>Sample rate</span><strong>{result.metrics.sample_rate_hz.toLocaleString()} Hz</strong></div>
          </div>
          <p>{result.message}</p>
          <small>{result.limitations.join(' ')}</small>
        </div>
      )}
      <small className="rir-note">This is signal analysis, not an authenticity prediction. RT60 requires a usable decay.</small>
    </section>
  )
}
