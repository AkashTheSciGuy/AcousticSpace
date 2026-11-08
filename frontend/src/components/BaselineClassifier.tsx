import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './BaselineClassifier.css'

type Status = { available: boolean; model_type: string; message: string }
type Prediction = {
  status: 'predicted'
  filename: string
  label: 'real' | 'fake'
  fake_probability: number
  real_probability: number
  disclaimer: string
}
const MAX_SIZE = 20 * 1024 * 1024

export default function BaselineClassifier() {
  const [status, setStatus] = useState<Status | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    controllerRef.current = controller
    void fetch('/api/model/status', { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`Model status unavailable (HTTP ${response.status}).`)
        const data: unknown = await response.json()
        if (!data || typeof data !== 'object' || typeof (data as Status).available !== 'boolean' ||
            typeof (data as Status).message !== 'string') throw new Error('Unexpected model status response.')
        setStatus(data as Status)
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Cannot reach model API.')
      })
    return () => controller.abort()
  }, [])

  function choose(event: ChangeEvent<HTMLInputElement>) {
    controllerRef.current?.abort()
    setBusy(false)
    setResult(null)
    setError(null)
    const selected = event.target.files?.[0] ?? null
    event.target.value = ''
    if (selected && (!/\.(wav|mp3)$/i.test(selected.name) || !selected.size || selected.size > MAX_SIZE)) {
      setFile(null)
      setError('Choose a nonempty WAV/MP3 recording under 20 MiB.')
    } else setFile(selected)
  }

  async function predict() {
    if (!file || !status?.available || busy) return
    const controller = new AbortController()
    controllerRef.current = controller
    setBusy(true)
    setResult(null)
    setError(null)
    const timeout = window.setTimeout(() => controller.abort(), 30000)
    try {
      const body = new FormData()
      body.append('file', file, file.name)
      const response = await fetch('/api/model/predict', { method: 'POST', body, signal: controller.signal })
      const data: unknown = await response.json()
      if (!response.ok) {
        const detail = data && typeof data === 'object' && 'detail' in data ? (data as { detail: unknown }).detail : null
        throw new Error(typeof detail === 'string' ? detail : `Prediction failed (HTTP ${response.status}).`)
      }
      if (!data || typeof data !== 'object') throw new Error('Invalid model response.')
      const prediction = data as Prediction
      if (prediction.status !== 'predicted' || !['real', 'fake'].includes(prediction.label) ||
          !Number.isFinite(prediction.fake_probability) || !Number.isFinite(prediction.real_probability) ||
          typeof prediction.disclaimer !== 'string') throw new Error('Unexpected prediction response.')
      if (!controller.signal.aborted) setResult(prediction)
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Prediction failed.')
      else setError('Prediction timed out or was cancelled.')
    } finally {
      window.clearTimeout(timeout)
      if (controllerRef.current === controller) { setBusy(false); controllerRef.current = null }
    }
  }

  return <section className="baseline-card" aria-labelledby="baseline-title">
    <span className="section-label">STEP 03 · EXPERIMENTAL CLASSIFICATION</span>
    <h2 id="baseline-title">Baseline model</h2>
    <p>Optional research-only audio classifier. It uses basic signal features, not RIR or AST.</p>
    <p className="baseline-status" role="status">
      {status ? (status.available ? 'Trained baseline available' : `Not ready: ${status.message}`) : 'Checking model availability…'}
    </p>
    <div className="baseline-controls">
      <input type="file" accept=".wav,.mp3,audio/wav,audio/mpeg" onChange={choose} aria-label="Choose recording for baseline classification" />
      <button type="button" onClick={() => { void predict() }} disabled={!status?.available || !file || busy}>
        {busy ? 'Analyzing…' : 'Run experimental prediction'}
      </button>
    </div>
    {file && <p>Selected: {file.name}</p>}
    {error && <p role="alert" className="baseline-error">{error}</p>}
    {result && <div className="baseline-results" role="status">
      <strong>Experimental prediction: {result.label.toUpperCase()}</strong>
      <p>Model fake-class score: {(result.fake_probability * 100).toFixed(1)}%</p>
      <small>{result.disclaimer}</small>
    </div>}
    <small>Class scores are not independently validated confidence estimates. No result is produced until a real trained model is installed.</small>
  </section>
}
