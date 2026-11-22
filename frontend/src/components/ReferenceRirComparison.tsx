import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './ReferenceRirComparison.css'

type Role = 'dry_speech' | 'wet_speech' | 'dry_ambience' | 'wet_ambience'
const ROLES: readonly { key: Role; label: string; help: string }[] = [
  { key: 'dry_speech', label: 'Original dry speech', help: 'Known voice signal before reverberation' },
  { key: 'wet_speech', label: 'Reverberant speech stem', help: 'Isolated reverberant voice, not the full recording' },
  { key: 'dry_ambience', label: 'Original dry ambience', help: 'Known original environmental audio' },
  { key: 'wet_ambience', label: 'Reverberant ambience stem', help: 'Isolated reverberant environment, not the full recording' },
]
const FEATURES = [
  { key: 'rt60_difference_seconds', label: 'RT60 difference', unit: ' s' },
  { key: 'rt60_missing', label: 'RT60 estimate missing', unit: '' },
  { key: 'c50_difference_db', label: 'C50 clarity difference', unit: ' dB' },
  { key: 'drr_difference_db', label: 'Direct/reverb ratio difference', unit: ' dB' },
  { key: 'late_energy_difference', label: 'Late-energy fraction difference', unit: '' },
  { key: 'decay_profile_distance_db', label: 'Decay-profile distance', unit: ' dB' },
  { key: 'spectral_shape_distance_db', label: 'Spectral-shape distance', unit: ' dB' },
] as const
type FeatureKey = typeof FEATURES[number]['key']
type Uploads = Record<Role, File | null>
type Report = {
  status: 'analyzed'
  input_type: 'four_reference_aided_audio_signals'
  sample_rate_hz: number
  prediction: null
  features: Record<FeatureKey, number>
  limitations: string[]
}
const MAX_BYTES = 4 * 1024 * 1024
const emptyFiles = (): Uploads => ({
  dry_speech: null, wet_speech: null, dry_ambience: null, wet_ambience: null,
})

function isReport(payload: unknown): payload is Report {
  if (!payload || typeof payload !== 'object') return false
  const result = payload as Partial<Report>
  if (result.status !== 'analyzed' || result.input_type !== 'four_reference_aided_audio_signals' ||
    result.sample_rate_hz !== 16000 || result.prediction !== null ||
    !Array.isArray(result.limitations) ||
    !result.limitations.every(item => typeof item === 'string') ||
    !result.features || typeof result.features !== 'object') return false
  return FEATURES.every(({ key }) => typeof result.features![key] === 'number' &&
    Number.isFinite(result.features![key]) && result.features![key] >= 0)
}

export default function ReferenceRirComparison() {
  const [files, setFiles] = useState<Uploads>(emptyFiles)
  const [report, setReport] = useState<Report | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const request = useRef<AbortController | null>(null)
  useEffect(() => () => { request.current?.abort() }, [])

  function choose(role: Role, event: ChangeEvent<HTMLInputElement>) {
    request.current?.abort()
    request.current = null
    setBusy(false)
    setReport(null)
    setError(null)
    const selected = event.target.files?.[0] ?? null
    event.target.value = ''
    if (selected && (!/\.wav$/i.test(selected.name) || selected.size === 0 ||
      selected.size > MAX_BYTES)) {
      setFiles(old => ({ ...old, [role]: null }))
      setError(role + ': choose a nonempty WAV no larger than 4 MiB.')
      return
    }
    setFiles(old => ({ ...old, [role]: selected }))
  }

  async function compare() {
    if (busy || ROLES.some(({ key }) => !files[key])) return
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    setBusy(true)
    setReport(null)
    setError(null)
    const timeout = window.setTimeout(() => controller.abort(), 45000)
    try {
      const body = new FormData()
      for (const { key } of ROLES) {
        const file = files[key]
        if (!file) throw new Error('Missing ' + key + ' input.')
        body.append(key, file, file.name)
      }
      const response = await fetch('/api/rir/reference-consistency', {
        method: 'POST', body, signal: controller.signal,
      })
      const data: unknown = await response.json()
      if (!response.ok) {
        const detail = data && typeof data === 'object' && 'detail' in data
          ? (data as { detail: unknown }).detail : null
        throw new Error(typeof detail === 'string' ? detail : 'RIR comparison failed (HTTP ' + response.status + ').')
      }
      if (!isReport(data)) throw new Error('Unexpected acoustic comparison response.')
      if (request.current === controller) setReport(data)
    } catch (cause) {
      if (request.current === controller) setError(controller.signal.aborted
        ? 'Room comparison timed out or was cancelled.'
        : cause instanceof Error ? cause.message : 'Room comparison failed.')
    } finally {
      window.clearTimeout(timeout)
      if (request.current === controller) {
        request.current = null
        setBusy(false)
      }
    }
  }

  const ready = ROLES.every(({ key }) => Boolean(files[key]))
  return (
    <section className="reference-rir-card" aria-labelledby="reference-rir-title">
      <span className="section-label">OPTIONAL CONTROLLED EXPERIMENT</span>
      <h2 id="reference-rir-title">Compare voice and ambience room responses</h2>
      <p>Supply <strong>four separately known/isolated WAV signals</strong>. This measures
        reference-aided acoustic differences, not authenticity, and does not extract
        room responses from a single mixed recording.</p>
      <div className="reference-rir-inputs">
        {ROLES.map(({ key, label, help }) => (
          <label key={key} className="reference-rir-input">
            <strong>{label}</strong>
            <span>{help}</span>
            <input type="file" accept=".wav,audio/wav" aria-label={label}
              onChange={event => choose(key, event)} />
            {files[key] && <small>Selected: {files[key]!.name}</small>}
          </label>
        ))}
      </div>
      <button type="button" className="reference-rir-submit" disabled={!ready || busy}
        onClick={() => { void compare() }}>
        {busy ? 'Comparing acoustic responses…' : 'Compare reference-aided responses'}
      </button>
      {error && <p className="reference-rir-error" role="alert">{error}</p>}
      {report && <div className="reference-rir-results" role="status">
        <h3>Acoustic differences (not a prediction)</h3>
        <dl className="reference-rir-metrics">
          {FEATURES.map(({ key, label, unit }) => (
            <div key={key}>
              <dt>{label}</dt>
              <dd>{key === 'rt60_missing'
                ? (report.features[key] === 1 ? 'Yes' : 'No')
                : report.features[key].toFixed(3) + unit}</dd>
            </div>
          ))}
        </dl>
        <p>No real/fake verdict or calibrated mismatch probability was produced.</p>
        <small>{report.limitations.join(' ')}</small>
      </div>}
      <small className="reference-rir-disclaimer">Each WAV must be mono 16 kHz,
        256 samples to 12 seconds, and at most 4 MiB. Do not provide the same
        mixed audio for every input.</small>
    </section>
  )
}
