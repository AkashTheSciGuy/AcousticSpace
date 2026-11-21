import { useEffect, useRef, useState } from 'react'
import './SegmentReview.css'

type ModelVariant = 'baseline' | 'acoustic'
type WindowScore = {
  start_seconds: number
  end_seconds: number
  fake_score: number
  flagged: boolean
}
type SegmentReport = {
  variant: ModelVariant
  warning: string
  segments: WindowScore[]
}
type Props = { file: File | null; variant: ModelVariant; available: boolean }

function isSegmentReport(data: unknown, variant: ModelVariant): data is SegmentReport {
  if (!data || typeof data !== 'object') return false
  const report = data as Partial<SegmentReport>
  return report.variant === variant && typeof report.warning === 'string' &&
    Array.isArray(report.segments) && report.segments.length > 0 &&
    report.segments.length <= 60 && report.segments.every((row) =>
      row && typeof row.start_seconds === 'number' && Number.isFinite(row.start_seconds) &&
      typeof row.end_seconds === 'number' && Number.isFinite(row.end_seconds) &&
      row.start_seconds >= 0 && row.end_seconds > row.start_seconds && row.end_seconds <= 60.001 &&
      typeof row.fake_score === 'number' && Number.isFinite(row.fake_score) &&
      row.fake_score >= 0 && row.fake_score <= 1 && typeof row.flagged === 'boolean')
}

export default function SegmentReview({ file, variant, available }: Props) {
  const [report, setReport] = useState<SegmentReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const activeRequest = useRef<AbortController | null>(null)

  useEffect(() => {
    activeRequest.current?.abort()
    activeRequest.current = null
    setReport(null)
    setError(null)
    setLoading(false)
    return () => { activeRequest.current?.abort(); activeRequest.current = null }
  }, [file, variant, available])

  async function review() {
    if (!file || !available || loading) return
    const controller = new AbortController()
    activeRequest.current = controller
    setLoading(true)
    setReport(null)
    setError(null)
    const timeout = window.setTimeout(() => controller.abort(), 30000)
    try {
      const body = new FormData()
      body.append('file', file, file.name)
      const response = await fetch('/api/model/segments?variant=' + variant, {
        method: 'POST', body, signal: controller.signal,
      })
      if (!response.ok) {
        let detail = 'Window analysis failed (HTTP ' + response.status + ').'
        try {
          const body: unknown = await response.json()
          if (body && typeof body === 'object' && 'detail' in body &&
              typeof body.detail === 'string') detail = body.detail
        } catch { /* The server may return a non-JSON error page. */ }
        throw new Error(detail)
      }
      const data: unknown = await response.json()
      if (!isSegmentReport(data, variant)) throw new Error('Unexpected window-analysis response.')
      if (activeRequest.current === controller) setReport(data)
    } catch (cause) {
      if (activeRequest.current === controller) {
        setError(controller.signal.aborted ? 'Window analysis timed out.' :
          cause instanceof Error ? cause.message : 'Window analysis failed.')
      }
    } finally {
      window.clearTimeout(timeout)
      if (activeRequest.current === controller) {
        activeRequest.current = null
        setLoading(false)
      }
    }
  }

  if (!file) return null
  return <div className="segment-review" aria-label="Experimental audio segment review">
    <button type="button" onClick={() => { void review() }} disabled={!available || loading}>
      {loading ? 'Scoring windows…' : 'Review audio windows'}
    </button>
    <small>Overlapping windows use the currently selected experimental classifier. This is not a validated tamper locator.</small>
    {error && <p role="alert">{error}</p>}
    {report && <div role="status">
      <h3>Window-level scores</h3>
      <div className="segment-list">
        {report.segments.map((row, index) => <div className="segment-entry" key={index}>
          <span>{row.start_seconds.toFixed(1)}–{row.end_seconds.toFixed(1)} s</span>
          <div className="segment-bar" aria-hidden="true">
            <span style={{ width: Math.round(row.fake_score * 100) + '%' }} />
          </div>
          <strong>{(row.fake_score * 100).toFixed(1)}%</strong>
          {row.flagged && <span className="segment-flag">Above 0.5 threshold</span>}
        </div>)}
      </div>
      <p className="segment-warning">{report.warning}</p>
    </div>}
  </div>
}
