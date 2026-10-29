import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'

// Temporary UI constraints: confirm formats and limits with the backend owner.
const SUPPORTED_EXTENSIONS = ['wav', 'mp3'] as const
const MAX_FILE_BYTES = 20 * 1024 * 1024

function formatFileSize(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function AudioUpload() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  function validateAndSelect(file: File): void {
    const extension = file.name.split('.').pop()?.toLowerCase()

    if (!extension || !SUPPORTED_EXTENSIONS.some((value) => value === extension)) {
      setError('Unsupported format. For this prototype, select a WAV or MP3 file.')
      setSelectedFile(null)
      return
    }

    if (file.size === 0) {
      setError('This file is empty. Select a recording that contains audio data.')
      setSelectedFile(null)
      return
    }

    if (file.size > MAX_FILE_BYTES) {
      setError('The selected file exceeds the current 20 MB prototype limit.')
      setSelectedFile(null)
      return
    }

    setSelectedFile(file)
    setError(null)
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0]
    if (file) validateAndSelect(file)
    // Enables selecting the same file again after clearing an error.
    event.target.value = ''
  }

  function handleDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files.item(0)
    if (file) validateAndSelect(file)
  }

  function clearSelection(): void {
    setSelectedFile(null)
    setError(null)
  }

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <div className="card-heading">
        <div>
          <span className="section-label">STEP 01</span>
          <h2 id="upload-title">Select an audio recording</h2>
          <p>Files are validated locally for now. No audio is uploaded to a server.</p>
        </div>
        <span className="local-badge">LOCAL ONLY</span>
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
        <small>Prototype formats: WAV, MP3 · Maximum: 20 MB</small>
      </div>

      {error && <p className="validation-error" role="alert">{error}</p>}

      {selectedFile && (
        <div className="selected-file" role="status" aria-live="polite">
          <div className="file-details">
            <span className="file-type" aria-hidden="true">AUDIO</span>
            <div>
              <strong>{selectedFile.name}</strong>
              <span>{formatFileSize(selectedFile.size)} · Ready for future API integration</span>
            </div>
          </div>
          <button className="text-button" type="button" onClick={clearSelection}>
            Remove
          </button>
        </div>
      )}

      <div className="upload-footer">
        <span>Backend analysis is not connected yet.</span>
        <button className="disabled-button" type="button" disabled title="API integration is a later milestone">
          Analyze audio (coming soon)
        </button>
      </div>
    </section>
  )
}
