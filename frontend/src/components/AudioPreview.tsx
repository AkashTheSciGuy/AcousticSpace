import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import './AudioPreview.css'

type AudioPreviewProps = {
  file: File
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const whole = Math.floor(seconds)
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}

/** Local waveform and playback controls; no recording is sent to the server. */
export default function AudioPreview({ file }: AudioPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const waveSurferRef = useRef<WaveSurfer | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    if (!containerRef.current) return

    let active = true
    setLoading(true)
    setError(null)
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)

    const wavesurfer = WaveSurfer.create({
      container: containerRef.current,
      height: 88,
      waveColor: '#627fae',
      progressColor: '#90adff',
      cursorColor: '#d9e4ff',
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      normalize: true,
    })

    waveSurferRef.current = wavesurfer
    const unsubscribers = [
      wavesurfer.on('ready', (total) => {
        if (!active) return
        setDuration(total)
        setLoading(false)
      }),
      wavesurfer.on('timeupdate', (elapsed) => {
        if (active) setCurrentTime(elapsed)
      }),
      wavesurfer.on('play', () => {
        if (active) setPlaying(true)
      }),
      wavesurfer.on('pause', () => {
        if (active) setPlaying(false)
      }),
      wavesurfer.on('finish', () => {
        if (active) setPlaying(false)
      }),
      wavesurfer.on('error', () => {
        if (active) {
          setError('The waveform could not be created. Try another audio file.')
          setLoading(false)
        }
      }),
    ]

    void wavesurfer.loadBlob(file).catch(() => {
      if (active) {
        setError('This recording cannot be decoded by the browser.')
        setLoading(false)
      }
    })

    return () => {
      active = false
      unsubscribers.forEach((unsubscribe) => unsubscribe())
      wavesurfer.destroy()
      if (waveSurferRef.current === wavesurfer) waveSurferRef.current = null
    }
  }, [file])

  function togglePlayback(): void {
    const player = waveSurferRef.current
    if (!player) return
    void player.playPause().catch(() => {
      setError('Audio playback failed. Check that this recording is supported.')
      setPlaying(false)
    })
  }

  return (
    <section className="audio-preview" aria-label="Local audio waveform and playback">
      <div className="audio-preview-heading">
        <strong>Audio waveform</strong>
        <span>Local preview · not uploaded</span>
      </div>

      <div
        ref={containerRef}
        className="audio-waveform"
        aria-label={`Waveform for ${file.name}. Click to seek through the audio.`}
      />

      {loading && !error && <p className="audio-preview-hint" role="status">Preparing waveform…</p>}
      {error && <p className="validation-error" role="alert">{error}</p>}

      <div className="audio-preview-controls">
        <button
          type="button"
          className="audio-play-button"
          onClick={togglePlayback}
          disabled={loading || Boolean(error)}
          aria-label={playing ? 'Pause audio' : 'Play audio'}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <label className="audio-seek-label">
          <span className="visually-hidden">Seek in audio</span>
          <input
            type="range"
            min={0}
            max={Math.max(duration, 1)}
            step={0.1}
            value={Math.min(currentTime, Math.max(duration, 1))}
            onChange={(event) => waveSurferRef.current?.setTime(Number(event.target.value))}
            disabled={loading || Boolean(error)}
          />
        </label>
        <span className="audio-time" aria-live="off">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
      <p className="audio-preview-hint">Click the waveform to seek, or use the slider with your keyboard.</p>
    </section>
  )
}
