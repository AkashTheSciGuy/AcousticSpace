import { useEffect, useRef, useState } from 'react'
import './SpeechDiagnostics.css'

type Interval = { start_seconds: number; end_seconds: number }
type Report = {
  status: 'analyzed'
  duration_seconds: number
  mel_bins: number
  time_bins: number
  log_mel_db: number[][]
  energy_peak_seconds: number[]
  energy_peak_rate_hz: number | null
  noisy_quiet_intervals: Interval[]
  warning: string
}

function validReport(data: unknown): data is Report {
  if (!data || typeof data !== 'object') return false
  const value = data as Partial<Report>
  return value.status === 'analyzed' && value.mel_bins === 48 &&
    typeof value.time_bins === 'number' && value.time_bins >= 1 &&
    value.time_bins <= 100 && Array.isArray(value.log_mel_db) &&
    value.log_mel_db.length === value.mel_bins &&
    value.log_mel_db.every(row => Array.isArray(row) && row.length === value.time_bins &&
      row.every(n => typeof n === 'number' && Number.isFinite(n) && n <= 0.01 && n >= -80.01)) &&
    Array.isArray(value.energy_peak_seconds) &&
    value.energy_peak_seconds.every(n => typeof n === 'number' && Number.isFinite(n)) &&
    Array.isArray(value.noisy_quiet_intervals) &&
    value.noisy_quiet_intervals.every(i => Number.isFinite(i.start_seconds) && Number.isFinite(i.end_seconds)) &&
    typeof value.warning === 'string'
}

export default function SpeechDiagnostics({ file }: { file: File }) {
  const [report, setReport] = useState<Report | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
    setReport(null)
    setLoading(false)
    setError(null)
    return () => { controllerRef.current?.abort(); controllerRef.current = null }
  }, [file])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !report) return
    const rows = report.mel_bins
    const columns = report.time_bins
    canvas.width = columns
    canvas.height = rows
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const pixels = ctx.createImageData(columns, rows)
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < columns; col++) {
        const intensity = Math.max(0, Math.min(1, (report.log_mel_db[row][col] + 80) / 80))
        const pixel = ((rows - row - 1) * columns + col) * 4
        pixels.data[pixel] = Math.round(25 + 225 * intensity)
        pixels.data[pixel + 1] = Math.round(35 + 145 * intensity)
        pixels.data[pixel + 2] = Math.round(80 + 110 * intensity)
        pixels.data[pixel + 3] = 255
      }
    }
    ctx.putImageData(pixels, 0, 0)
  }, [report])

  async function run() {
    if (loading) return
    const controller = new AbortController()
    controllerRef.current?.abort()
    controllerRef.current = controller
    setLoading(true)
    setError(null)
    setReport(null)
    const timeout = window.setTimeout(() => controller.abort(), 30000)
    try {
      const body = new FormData()
      body.append('file', file, file.name)
      const response = await fetch('/api/audio/diagnostics', {
        method: 'POST', body, signal: controller.signal,
      })
      const payload: unknown = await response.json()
      if (!response.ok) {
        const detail = payload && typeof payload === 'object' && 'detail' in payload &&
          typeof payload.detail === 'string' ? payload.detail : 'Diagnostics failed.'
        throw new Error(detail)
      }
      if (!validReport(payload)) throw new Error('Unexpected diagnostic response.')
      if (controllerRef.current === controller) setReport(payload)
    } catch (cause) {
      if (controllerRef.current === controller) setError(
        controller.signal.aborted ? 'Diagnostics timed out.' :
        cause instanceof Error ? cause.message : 'Diagnostics failed.')
    } finally {
      window.clearTimeout(timeout)
      if (controllerRef.current === controller) {
        setLoading(false)
        controllerRef.current = null
      }
    }
  }

  return <div className="speech-diagnostics" aria-label="Speech spectrogram and cadence inspection">
    <button type="button" onClick={() => { void run() }} disabled={loading}>
      {loading ? 'Analyzing spectrogram…' : 'Inspect spectrogram and cadence'}
    </button>
    {error && <p role="alert">{error}</p>}
    {report && <div role="status">
      <h3>Log-mel spectrogram</h3>
      <canvas className="speech-spectrogram" ref={canvasRef} role="img"
        aria-label="Approximate log-mel spectrogram of the selected recording, frequency increasing upward" />
      <p>Energy peaks: {report.energy_peak_seconds.length}
        {report.energy_peak_rate_hz !== null ? ' · Estimated peak rate: ' + report.energy_peak_rate_hz.toFixed(2) + '/s' : ''}
      </p>
      <p>Quiet noise-like intervals: {report.noisy_quiet_intervals.length}</p>
      {report.noisy_quiet_intervals.length > 0 && <ol>
        {report.noisy_quiet_intervals.map((item, i) => <li key={i}>
          {item.start_seconds.toFixed(2)}–{item.end_seconds.toFixed(2)} s
        </li>)}
      </ol>}
      <small>{report.warning}</small>
    </div>}
    <small>Energy peaks are not aligned syllables. Quiet intervals can be background noise, not breathing.</small>
  </div>
}
