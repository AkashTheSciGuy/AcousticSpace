import { useEffect, useRef, useState } from 'react'
import './AcousticComparison.css'

type Metric = {
  samples: number
  accuracy: number
  precision_fake: number
  recall_fake: number
  f1_fake: number
}
type Comparison = {
  experiment: 'speech_acoustics_feature_ablation_v1'
  warning: string
  recordings: { train: number; val: number; test: number }
  models: {
    baseline_three_features: { metrics: { val: Metric; test: Metric } }
    speech_acoustic_six_features: { metrics: { val: Metric; test: Metric } }
  }
}
const modelRows = [
  { key: 'baseline_three_features', label: '3-feature baseline' },
  { key: 'speech_acoustic_six_features', label: '6-feature candidate' },
] as const

function validComparison(data: unknown): data is Comparison {
  if (!data || typeof data !== 'object') return false
  const value = data as Partial<Comparison>
  if (value.experiment !== 'speech_acoustics_feature_ablation_v1' ||
      typeof value.warning !== 'string' || !value.models) return false
  return modelRows.every(({ key }) => {
    const metrics = value.models?.[key]?.metrics
    return metrics && (['val', 'test'] as const).every((split) => {
      const item = metrics[split]
      return item && Number.isInteger(item.samples) && item.samples > 0 &&
        (['accuracy', 'precision_fake', 'recall_fake', 'f1_fake'] as const).every(
          (metric) => Number.isFinite(item[metric]) && item[metric] >= 0 && item[metric] <= 1,
        )
    })
  })
}

export default function AcousticComparison() {
  const [report, setReport] = useState<Comparison | null>(null)
  const [state, setState] = useState<'loading' | 'ready' | 'missing' | 'error'>('loading')
  const controllerRef = useRef<AbortController | null>(null)

  async function loadReport(controller: AbortController): Promise<void> {
    controllerRef.current?.abort()
    controllerRef.current = controller
    setState('loading')
    try {
      const response = await fetch('/api/model/comparison', { signal: controller.signal })
      if (response.status === 404) {
        if (!controller.signal.aborted) { setReport(null); setState('missing') }
        return
      }
      if (!response.ok) throw new Error('Comparison API unavailable')
      const data: unknown = await response.json()
      if (!validComparison(data)) throw new Error('Invalid comparison report')
      if (!controller.signal.aborted) { setReport(data); setState('ready') }
    } catch {
      if (!controller.signal.aborted) { setReport(null); setState('error') }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    void loadReport(controller)
    return () => controller.abort()
  }, [])

  return (
    <section className="acoustic-compare" aria-labelledby="acoustic-compare-title">
      <div className="acoustic-compare-heading">
        <div>
          <span className="section-label">EXPERIMENTAL RESEARCH</span>
          <h2 id="acoustic-compare-title">Feature comparison</h2>
        </div>
        <button type="button" disabled={state === 'loading'}
          onClick={() => { void loadReport(new AbortController()) }}>
          {state === 'loading' ? 'Loading…' : 'Refresh comparison'}
        </button>
      </div>
      <p>Compare the original three-feature baseline against a six-feature speech-acoustic model on identical labeled splits.</p>
      {state === 'loading' && <p role="status">Checking for a local comparison report…</p>}
      {state === 'missing' && <p role="status">No comparison results yet. Run the backend acoustic comparison experiment on your local dataset, then refresh.</p>}
      {state === 'error' && <p role="alert">Could not load a valid comparison report. Check the backend and retry.</p>}
      {state === 'ready' && report && (
        <>
          <div className="acoustic-table-scroll">
            <table>
              <caption>Measured experiment metrics; higher scores do not establish real-world accuracy</caption>
              <thead>
                <tr><th scope="col">Model</th><th scope="col">Split</th><th scope="col">Accuracy</th><th scope="col">Fake F1</th><th scope="col">Fake recall</th></tr>
              </thead>
              <tbody>
                {modelRows.flatMap(({ key, label }) => (['val', 'test'] as const).map((split) => {
                  const metrics = report.models[key].metrics[split]
                  return (
                    <tr key={`${key}-${split}`}>
                      <th scope="row">{label}</th>
                      <td>{split === 'val' ? 'Validation' : 'Test'} ({metrics.samples})</td>
                      <td>{(metrics.accuracy * 100).toFixed(1)}%</td>
                      <td>{(metrics.f1_fake * 100).toFixed(1)}%</td>
                      <td>{(metrics.recall_fake * 100).toFixed(1)}%</td>
                    </tr>
                  )
                }))}
              </tbody>
            </table>
          </div>
          <small>{report.warning}</small>
        </>
      )}
      <small>These features characterize recorded speech, not a measured room impulse response. Neither model automatically replaces the deployed baseline.</small>
    </section>
  )
}
