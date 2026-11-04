import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import AudioPreview from './AudioPreview'
import { validateAudio, validationMode } from '../services/audioApi'
import type { AudioValidationResult } from '../services/audioApi'
import './AudioUpload.css'

// Current prototype limits; the server validates the content again in live mode.
const SUPPORTED_EXTENSIONS = ['wav', 'mp3'] as const
const MAX_FILE_BYTES = 20 * 1024 * 1024
const REQUEST_TIMEOUT_MS = 15_000

type ValidationState = 'idle' | 'validating' | 'success' | 'error'

function formatFileSize(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function AudioUpload() {
  const inputRef = useRef<HTMLInputElement>(null)
  const requestRef = useRef<AbortController | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [validationState, setValidationState] = useState<ValidationState>('idle')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [validationResult, setValidationResult] = useState<AudioValidationResult | null>(null)

  useEffect(() => () => requestRef.current?.abort(), [])

  function resetValidation(): void {
    requestRef.current?.abort()
    requestRef.current = null
    setValidationState('idle')
    setValidationError(null)
    setValidationResult(null)
  }

  function validateAndSelect(file: File): void {
    resetValidation()
    const extension = file.name.split('.').pop()?.toLowerCase()
    if (!extension || !SUPPORTED_EXTENSIONS.some((value) => value === extension)) {
      setFileError('Unsupported format. Select a WAV or MP3 file.')
      setSelectedFile(null)
      return
    }
    if (file.size === 0) {
      setFileError('This file is empty. Select a recording that contains audio.')
      setSelectedFile(null)
      return
    }
    if (file.size > MAX_FILE_BYTES) {
      setFileError('The selected file exceeds the 20 MiB limit.')
      setSelectedFile(null)
      return
    }
    setSelectedFile(file)
    setFileError(null)
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0]
    if (file) validateAndSelect(file)
    event.target.value = ''
  }

  function handleDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files.item(0)
    if (file) validateAndSelect(file)
  }

  function clearSelection(): void {
    resetValidation()
    setSelectedFile(null)
    setFileError(null)
  }

  async function handleValidate(): Promise<void> {
    if (!selectedFile || validationState === 'validating') return

    const controller = new AbortController()
    requestRef.current = controller
    let timedOut = false
    const timeout = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, REQUEST_TIMEOUT_MS)
    setValidationState('validating')
    setValidationResult(null)
    setValidationError(null)

    try {
      const result = await validateAudio(selectedFile, controller.signal)
      if (requestRef.current !== controller) return
      setValidationResult(result)
      setValidationState('success')
    } catch (error) {
      if (requestRef.current !== controller) return
      if (controller.signal.aborted && !timedOut) return
      setValidationError(
        timedOut ? 'The request timed out. Please try again.'
          : error instanceof Error ? error.message : 'Audio validation failed.',
      )
      setValidationState('error')
    } finally {
      window.clearTimeout(timeout)
      if (requestRef.current === controller) requestRef.current = null
    }
  }

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <div className="card-heading">
        <div>
          <span className="section-label">STEP 01</span>
          <h2 id="upload-title">Select an audio recording</h2>
          <p>Choose a recording, inspect its waveform, then inspect its signal characteristics.</p>
        </div>
        <span className="local-badge">{validationMode === 'mock' ? 'MOCK API' : 'LIVE API'}</span>
      </div>

      <div
        className={`drop-zone${isDragging ? ' is-dragging' : ''}`}
        onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <div className="upload-icon" aria-hidden="true">↑</div>
        <h3>Drop your recording here</h3>
        <p>Or choose a file from your device.</p>
        <button className="primary-button" type="button" onClick={() => inputRef.current?.click()}>
          Browse audio files
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,audio/wav,audio/mpeg"
          onChange={handleInputChange}
          className="visually-hidden"
          aria-label="Choose a WAV or MP3 audio file"
        />
        <small>WAV or MP3 · Maximum 20 MiB</small>
      </div>

      {fileError && <p className="validation-error" role="alert">{fileError}</p>}

      {selectedFile && (
        <>
          <div className="selected-file" role="status">
            <div className="file-details">
              <span className="file-type" aria-hidden="true">AUDIO</span>
              <div>
                <strong>{selectedFile.name}</strong>
                <span>{formatFileSize(selectedFile.size)} · Ready to validate</span>
              </div>
            </div>
            <button className="text-button" type="button" onClick={clearSelection}>Remove</button>
          </div>
          <AudioPreview file={selectedFile} />
        </>
      )}

      <div className="upload-footer">
        <span>{validationMode === 'mock'
          ? 'Demo mode: no file is sent to a server.'
          : 'Live mode: FastAPI measures audio features; it does not detect deepfakes.'}</span>
        <button
          className="validation-action-button"
          type="button"
          disabled={!selectedFile || validationState === 'validating'}
          onClick={() => { void handleValidate() }}
        >
          {validationState === 'validating' ? 'Inspecting…'
            : validationState === 'error' ? 'Retry inspection' : 'Inspect recording'}
        </button>
      </div>

      {validationState === 'validating' && (
        <p className="validation-progress" role="status">Inspecting recording…</p>
      )}
      {validationState === 'error' && validationError && (
        <p className="validation-error" role="alert">{validationError}</p>
      )}
      {validationState === 'success' && validationResult && (
        <div className="validation-success" role="status">
          <strong>Audio inspection {validationMode === 'mock' ? '(simulated)' : 'complete'}</strong>
          <span>{validationResult.filename} · {validationResult.format.toUpperCase()} · {formatFileSize(validationResult.size_bytes)}</span>
          <p>{validationResult.message}</p>
          {validationResult.features && (
            <div className="audio-feature-grid" aria-label="Measured audio features">
              <div><span>Duration</span><strong>{validationResult.features.duration_seconds.toFixed(2)} s</strong></div>
              <div><span>Sample rate</span><strong>{validationResult.features.sample_rate.toLocaleString()} Hz</strong></div>
              <div><span>RMS amplitude</span><strong>{validationResult.features.rms_amplitude.toFixed(4)}</strong></div>
              <div><span>Zero-crossing rate</span><strong>{validationResult.features.zero_crossing_rate.toFixed(4)}</strong></div>
              <div><span>Spectral centroid</span><strong>{validationResult.features.spectral_centroid_hz.toFixed(1)} Hz</strong></div>
            </div>
          )}
          <small>Signal measurements only — not a deepfake-detection result.</small>
        </div>
      )}
    </section>
  )
}
