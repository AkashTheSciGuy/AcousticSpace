import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './BaselineClassifier.css'

type EvaluationMetrics = {
  samples: number
  accuracy: number
  precision_fake: number
  recall_fake: number
  f1_fake: number
  confusion_matrix_real_fake: number[][]
}
type Evaluation = {
  source: 'local_training_report'
  warning: string
  metrics: { val: EvaluationMetrics; test: EvaluationMetrics }
}
type Status = {
  available: boolean
  model_type: string
  message: string
  evaluation?: Evaluation
}
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
  const [statusLoading, setStatusLoading] = useState(true)
  const statusControllerRef = useRef<AbortController | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)

  async function refreshStatus(controller: AbortController): Promise<void> {
    statusControllerRef.current?.abort()
    statusControllerRef.current = controller
    setStatusLoading(true)
    try {
      const response = await fetch('/api/model/status', { signal: controller.signal })
      if (!response.ok) throw new Error(`Model status unavailable (HTTP ${response.status}).`)
      const data: unknown = await response.json()
      if (!data || typeof data !== 'object' || typeof (data as Status).available !== 'boolean' ||
          typeof (data as Status).message !== 'string') throw new Error('Unexpected model status response.')
      if (!controller.signal.aborted) {
        setStatus(data as Status)
        setResult(null)
        setError(null)
      }
    } catch (cause: unknown) {
      if (!controller.signal.aborted) {
        setStatus(null)
        setError(cause instanceof Error ? cause.message : 'Cannot reach model API.')
      }
    } finally {
      if (statusControllerRef.current === controller) {
        setStatusLoading(false)
        statusControllerRef.current = null
      }
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    void refreshStatus(controller)
    return () => {
      controller.abort()
      controllerRef.current?.abort()
    }
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
    <div className="baseline-status-row">
      <p className="baseline-status" role="status">
        {status ? (status.available ? 'Trained baseline available' : `Not ready: ${status.message}`)
          : statusLoading ? 'Checking model availability…' : 'Model status could not be loaded.'}
      </p>
      <button type="button" className="baseline-refresh" disabled={statusLoading}
        onClick={() => { void refreshStatus(new AbortController()) }}>
        {statusLoading ? 'Checking…' : 'Refresh status'}
      </button>
    </div>
    {status?.available && status.evaluation && (
      <div className="baseline-evaluation" aria-label="Saved baseline evaluation">
        <h3>Saved experiment results</h3>
        <div className="baseline-evaluation-grid">
          {(['val', 'test'] as const).map(split => {
            const metrics = status.evaluation!.metrics[split]
            return <div key={split} className="baseline-evaluation-split">
              <strong>{split === 'val' ? 'Validation' : 'Test'} split</strong>
              <span>Samples: {metrics.samples}</span>
              <span>Accuracy: {(metrics.accuracy * 100).toFixed(1)}%</span>
              <span>Fake precision: {(metrics.precision_fake * 100).toFixed(1)}%</span>
              <span>Fake recall: {(metrics.recall_fake * 100).toFixed(1)}%</span>
              <span>Fake-class F1: {(metrics.f1_fake * 100).toFixed(1)}%</span>
              <div className="baseline-matrix">
                <span>Confusion matrix (actual × predicted)</span>
                <table aria-label={`${split} confusion matrix`}>
                  <thead><tr><th scope="col">Actual / Predicted</th><th scope="col">Real</th><th scope="col">Fake</th></tr></thead>
                  <tbody>
                    <tr><th scope="row">Real</th><td>{metrics.confusion_matrix_real_fake[0][0]}</td><td>{metrics.confusion_matrix_real_fake[0][1]}</td></tr>
                    <tr><th scope="row">Fake</th><td>{metrics.confusion_matrix_real_fake[1][0]}</td><td>{metrics.confusion_matrix_real_fake[1][1]}</td></tr>
                  </tbody>
                </table>
              </div>
              <small className="baseline-confusion-help">
                False positive: a real recording predicted fake. False negative: a fake recording predicted real.
                Lower false-negative counts mean fewer fake clips were missed on this split.
              </small>
            </div>
          })}
        </div>
        <p>{status.evaluation.warning}</p>
      </div>
    )}
    {!status?.available && !statusLoading && (
      <div className="baseline-training-guide" aria-label="How to enable baseline predictions">
        <h3>To enable experimental predictions</h3>
        <ol>
          <li>Collect correctly labeled real and AI-generated recordings.</li>
          <li>Prepare speaker/source-independent train, validation and test splits.</li>
          <li>Train the baseline locally and review its evaluation results.</li>
          <li>Install the trusted model artifact, then refresh model status.</li>
        </ol>
        <p className="baseline-dataset-reference">
          Suggested research dataset:{' '}
          <a href="https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45"
             target="_blank" rel="noopener noreferrer">
            ASVspoof 2019 Logical Access (official source)
          </a>. Download and prepare the dataset locally; it is not included in the app.
        </p>
        <small>No trained model is bundled with AcousticSpace. The baseline does not analyze RIR.</small>
      </div>
    )}
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
