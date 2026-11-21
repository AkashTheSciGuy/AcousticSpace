/** Session-only, metadata-only history. Never persist raw audio or filenames. */
export type AnalysisHistoryItem = {
  id: string
  timestamp: string
  variant: 'baseline' | 'acoustic'
  label: 'real' | 'fake'
  fakeScore: number
}

const KEY = 'acousticspace:session-history:v1'
const LIMIT = 20
export const HISTORY_CHANGED_EVENT = 'acousticspace:history-changed'

function isItem(value: unknown): value is AnalysisHistoryItem {
  if (!value || typeof value !== 'object') return false
  const v = value as Partial<AnalysisHistoryItem>
  return typeof v.id === 'string' && typeof v.timestamp === 'string' &&
    (v.variant === 'baseline' || v.variant === 'acoustic') &&
    (v.label === 'real' || v.label === 'fake') &&
    typeof v.fakeScore === 'number' && Number.isFinite(v.fakeScore) &&
    v.fakeScore >= 0 && v.fakeScore <= 1
}

export function readAnalysisHistory(): AnalysisHistoryItem[] {
  try {
    const raw = window.sessionStorage.getItem(KEY)
    if (!raw) return []
    const decoded: unknown = JSON.parse(raw)
    return Array.isArray(decoded) ? decoded.filter(isItem).slice(0, LIMIT) : []
  } catch { return [] }
}

export function addAnalysisHistory(item: Pick<AnalysisHistoryItem, 'variant' | 'label' | 'fakeScore'>): void {
  if (!Number.isFinite(item.fakeScore) || item.fakeScore < 0 || item.fakeScore > 1) return
  try {
    const entry: AnalysisHistoryItem = {
      ...item,
      id: window.crypto.randomUUID(),
      timestamp: new Date().toISOString(),
    }
    window.sessionStorage.setItem(KEY, JSON.stringify([entry, ...readAnalysisHistory()].slice(0, LIMIT)))
    window.dispatchEvent(new Event(HISTORY_CHANGED_EVENT))
  } catch { /* Browsers may disable session storage. Predictions still work. */ }
}

export function clearAnalysisHistory(): void {
  try { window.sessionStorage.removeItem(KEY) } catch { /* No persistent state to clear. */ }
  window.dispatchEvent(new Event(HISTORY_CHANGED_EVENT))
}
