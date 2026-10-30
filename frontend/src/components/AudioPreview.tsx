import { useEffect, useState } from 'react'
import './AudioPreview.css'

type AudioPreviewProps = {
  file: File
}

/** A local preview; the recording is not sent to the backend. */
export default function AudioPreview({ file }: AudioPreviewProps) {
  const [source, setSource] = useState<string | null>(null)
  const [playbackError, setPlaybackError] = useState(false)

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file)
    setSource(objectUrl)
    setPlaybackError(false)

    // Release browser memory when the user removes or replaces the file.
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])

  return (
    <div className="audio-preview" aria-label="Local audio preview">
      <div className="audio-preview-heading">
        <strong>Listen to your recording</strong>
        <span>Preview only · not uploaded</span>
      </div>

      {source && (
        <audio
          className="audio-preview-player"
          controls
          preload="metadata"
          src={source}
          aria-label={`Audio preview: ${file.name}`}
          onError={() => setPlaybackError(true)}
        >
          Your browser does not support audio playback.
        </audio>
      )}

      {playbackError && (
        <p className="validation-error" role="alert">
          This recording could not be played in your browser. Try a different audio file.
        </p>
      )}
    </div>
  )
}
