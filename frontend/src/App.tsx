import AudioUpload from './components/AudioUpload'
import './App.css'

const pipelineSteps = [
  { number: '01', title: 'Audio intake', description: 'Choose and validate a recording.', status: 'In progress' },
  { number: '02', title: 'Waveform preview', description: 'Visualize the selected audio.', status: 'Planned' },
  { number: '03', title: 'Acoustic analysis', description: 'Send the clip to the FastAPI model.', status: 'Planned' },
  { number: '04', title: 'Evidence review', description: 'Review the model output and its limits.', status: 'Planned' },
] as const

function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Project information">
        <div className="brand">
          <span className="brand-symbol" aria-hidden="true">◎</span>
          <div>
            <strong>AcousticSpace</strong>
            <span>Audio forensics workspace</span>
          </div>
        </div>

        <div className="sidebar-section">
          <span className="section-label">WORKSPACE</span>
          <div className="active-page" aria-current="page">
            <span className="page-dot" aria-hidden="true" />
            Analysis dashboard
          </div>
        </div>

        <div className="sidebar-footer">
          <span className="environment-dot" aria-hidden="true" />
          Frontend prototype
          <small>Model inference not connected</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <span>AcousticSpace / Dashboard</span>
          <span className="version">Prototype v0.1</span>
        </header>

        <section className="intro" aria-labelledby="page-title">
          <div className="eyebrow"><span aria-hidden="true" /> AUDIO FORENSICS</div>
          <h1 id="page-title">Analyze the space behind the sound.</h1>
          <p>
            AcousticSpace explores whether a voice recording's acoustic environment
            contains evidence of manipulation. Start by selecting an audio file.
          </p>
        </section>

        <AudioUpload />

        <section className="info-grid" aria-label="Project information">
          <article className="info-card">
            <div className="info-card-icon" aria-hidden="true">01</div>
            <h2>Room acoustics</h2>
            <p>
              Future analysis will examine reverberation-related features. Any detection
              claim must be supported by model evaluation.
            </p>
          </article>
          <article className="info-card">
            <div className="info-card-icon" aria-hidden="true">02</div>
            <h2>Explainable results</h2>
            <p>
              Predictions and confidence will appear only when the backend provides
              validated results and supporting fields.
            </p>
          </article>
        </section>
      </main>

      <aside className="right-panel" aria-label="Development progress">
        <span className="section-label">DEVELOPMENT ROADMAP</span>
        <h2>Analysis pipeline</h2>
        <p className="panel-caption">Working toward a complete upload-to-result flow.</p>
        <ol className="pipeline-list">
          {pipelineSteps.map((step) => (
            <li key={step.number} className="pipeline-step">
              <span className="step-number">{step.number}</span>
              <div>
                <h3>{step.title}</h3>
                <p>{step.description}</p>
                <span className={step.status === 'In progress' ? 'step-status progress' : 'step-status'}>
                  {step.status}
                </span>
              </div>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  )
}

export default App
