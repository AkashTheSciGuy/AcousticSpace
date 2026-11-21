import { useEffect, useRef, useState } from 'react'
import './AstClassifier.css'

type Status = { available: boolean; model_type: string; message: string }
type Prediction = { status: 'predicted'; label: 'real' | 'fake'; fake_probability: number;
  analyzed_seconds: number; disclaimer: string }

export default function AstClassifier() {
  const [status, setStatus] = useState<Status | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const statusAbort = useRef<AbortController | null>(null)
  const requestAbort = useRef<AbortController | null>(null)

  async function refresh() {
    statusAbort.current?.abort()
    const controller = new AbortController()
    statusAbort.current = controller
    setError(null)
    setResult(null)
    try {
      const response = await fetch('/api/ast/status', { signal: controller.signal })
      const data: unknown = await response.json()
      if (!response.ok || !data || typeof data !== 'object' ||
          typeof (data as Status).available !== 'boolean' ||
          typeof (data as Status).message !== 'string') throw new Error('Cannot read AST status.')
      if (statusAbort.current === controller) setStatus(data as Status)
    } catch (cause) {
      if (statusAbort.current === controller) setError(
        cause instanceof Error ? cause.message : 'AST status unavailable.')
    } finally {
      if (statusAbort.current === controller) statusAbort.current = null
    }
  }

  useEffect(() => {
    void refresh()
    return () => { statusAbort.current?.abort(); requestAbort.current?.abort() }
  }, [])

  async function predict() {
    if (!file || !status?.available || busy) return
    requestAbort.current?.abort()
    const controller = new AbortController()
    requestAbort.current = controller
    setBusy(true)
    setResult(null)
    setError(null)
    const timeout = window.setTimeout(() => controller.abort(), 90000)
    try {
      const body = new FormData()
      body.append('file', file, file.name)
      const response = await fetch('/api/ast/predict', {
        method: 'POST', body, signal: controller.signal,
      })
      const data: unknown = await response.json()
      if (!response.ok) {
        const message = data && typeof data === 'object' && 'detail' in data &&
          typeof data.detail === 'string' ? data.detail : 'AST prediction request failed.'
        throw new Error(message)
      }
      if (!data || typeof data !== 'object') throw new Error('Unexpected AST response.')
      const value = data as Prediction
      if (value.status !== 'predicted' || !['real', 'fake'].includes(value.label) ||
          !Number.isFinite(value.fake_probability) || value.fake_probability < 0 ||
          value.fake_probability > 1 || !Number.isFinite(value.analyzed_seconds) ||
          typeof value.disclaimer !== 'string') throw new Error('Unexpected AST prediction values.')
      if (requestAbort.current === controller) setResult(value)
    } catch (cause) {
      if (requestAbort.current === controller) setError(
        controller.signal.aborted ? 'AST prediction timed out.' :
        cause instanceof Error ? cause.message : 'AST inference failed.')
    } finally {
      window.clearTimeout(timeout)
      if (requestAbort.current === controller) { requestAbort.current = null; setBusy(false) }
    }
  }

  return <section className="ast-card" aria-labelledby="ast-title">
    <h2 id="ast-title">Audio Spectrogram Transformer (optional experiment)</h2>
    <p>The pretrained AST needs a locally trained real/fake checkpoint before predictions can run. It does not measure a room impulse response.</p>
    <div className="ast-status">
      <span role="status">{status ? (status.available ? 'Trained AST available' : 'Unavailable: ' + status.message) : 'Checking AST readiness…'}</span>
      <button type="button" onClick={() => { void refresh() }}>Refresh AST status</button>
    </div>
    <input type="file" accept=".wav,.mp3,audio/wav,audio/mpeg" aria-label="Choose AST recording"
      onChange={(event) => {
        requestAbort.current?.abort()
        requestAbort.current = null
        setBusy(false)
        setResult(null)
        setError(null)
        const selected = event.target.files?.[0] ?? null
        event.target.value = ''
        if (selected && (!/\.(wav|mp3)$/i.test(selected.name) ||
            selected.size === 0 || selected.size > 20 * 1024 * 1024)) {
          setFile(null)
          setError('Choose a nonempty WAV/MP3 recording under 20 MiB.')
        } else setFile(selected)
      }} />
    <button type="button" disabled={!status?.available || !file || busy} onClick={() => { void predict() }}>
      {busy ? 'Running AST…' : 'Run optional AST prediction'}
    </button>
    {error && <p role="alert">{error}</p>}
    {result && <div role="status">
      <strong>AST experiment: {result.label.toUpperCase()}</strong>
      <p>Fake-class softmax score: {(result.fake_probability * 100).toFixed(1)}%</p>
      <small>Analyzed {result.analyzed_seconds.toFixed(1)} seconds. {result.disclaimer}</small>
    </div>}
    <small>Optional dependencies and locally trained safetensors model required. No automatic AST download or training occurs during web requests.</small>
  </section>
}
