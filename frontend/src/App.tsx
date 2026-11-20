import AudioUpload from './components/AudioUpload'
import RirAnalyzer from './components/RirAnalyzer'
import BaselineClassifier from './components/BaselineClassifier'
import AcousticComparison from './components/AcousticComparison'
import './App.css'

const pipelineSteps = [
  { number: '01', title: 'Audio intake', description: 'Choose and validate a recording.', status: 'Working' },
  { number: '02', title: 'Waveform preview', description: 'Visualize the selected audio.', status: 'Working' },
  { number: '03', title: 'Acoustic analysis', description: 'Measure audio features and analyze an uploaded RIR.', status: 'Working' },
  { number: '04', title: 'Experimental inference', description: 'Baseline and optional speech-acoustic candidate trained locally; model artifacts required on each device.', status: 'Working' },
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
          Local research prototype
          <small>Trained model artifacts are stored locally, not in GitHub</small>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar">
          <span>AcousticSpace / Dashboard</span>
          <span className="version">Research prototype · Day 20</span>
        </header>
        <section className="intro" aria-labelledby="page-title">
          <div className="eyebrow"><span aria-hidden="true" /> AUDIO FORENSICS</div>
          <h1 id="page-title">Analyze the space behind the sound.</h1>
          <p>
            Inspect speech features, run experimental speech-acoustic classifiers, or analyze
            a separately measured room impulse response. These classifiers do not estimate RIR.
          </p>
        </section>
        <AudioUpload />
        <RirAnalyzer />
        <BaselineClassifier />
        <AcousticComparison />
        <section className="info-grid" aria-label="Project information">
          <article className="info-card">
            <div className="info-card-icon" aria-hidden="true">01</div>
            <h2>Room acoustics</h2>
            <p>RIR analysis measures RT60, clarity and approximate direct/reverberant energy ratio from impulse-response files, not ordinary speech.</p>
          </article>
          <article className="info-card">
            <div className="info-card-icon" aria-hidden="true">02</div>
            <h2>Explainable results</h2>
            <p>Local experimental predictions are available for trained models. Class scores are not calibrated confidence and do not establish forensic authenticity.</p>
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
                <span className={step.status === 'Working' ? 'step-status progress' : 'step-status'}>{step.status}</span>
              </div>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  )
}

export default App
