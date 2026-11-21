import { useEffect, useState } from 'react'
import { clearAnalysisHistory, HISTORY_CHANGED_EVENT, readAnalysisHistory } from '../services/analysisHistory'
import './HistoryPanel.css'

export default function HistoryPanel() {
  const [entries, setEntries] = useState(readAnalysisHistory)
  useEffect(() => {
    const refresh = () => setEntries(readAnalysisHistory())
    window.addEventListener(HISTORY_CHANGED_EVENT, refresh)
    return () => window.removeEventListener(HISTORY_CHANGED_EVENT, refresh)
  }, [])

  return <section className="history-panel" aria-labelledby="history-title">
    <div className="history-heading">
      <h2 id="history-title">Session analysis history</h2>
      <button type="button" onClick={clearAnalysisHistory} disabled={entries.length === 0}>
        Clear history
      </button>
    </div>
    <p>Only model labels, numeric scores and time are stored in this browser tab's session. Audio files and filenames are never saved.</p>
    {entries.length === 0 ? <p>No predictions recorded in this session.</p> :
      <ol className="history-items">
        {entries.map((entry) => <li key={entry.id}>
          <span>{new Date(entry.timestamp).toLocaleString()}</span>
          <strong>{entry.variant === 'baseline' ? '3-feature' : '6-feature'} · {entry.label.toUpperCase()}</strong>
          <span>Fake-class score: {(entry.fakeScore * 100).toFixed(1)}%</span>
        </li>)}
      </ol>}
    <small>These scores are experimental and are not calibrated confidence estimates.</small>
  </section>
}
