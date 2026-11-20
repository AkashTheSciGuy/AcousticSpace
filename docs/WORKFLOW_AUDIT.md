# AcousticSpace workflow audit

Audit date: 20 November 2026. Reference: **AcousticSpace Workflow.docx** in the repository and the original supplied document. This is an implementation comparison, not research validation.

| Original requirement | Implementation and evidence | Remaining limitation |
| --- | --- | --- |
| Python/FastAPI and reproducible setup | Application routes, pytest suite, Windows instructions, Python 3.12 CI | Dependencies use ranges; existing pickle/joblib models require compatible training dependencies |
| Labeled real/fake dataset, held-out evaluation | ASVspoof LA importer; hash, declared source-group separation and both-class checks; baseline/candidate training and evaluation scripts | No dataset, manifest, original trained artifacts or original experiment reports were available in this checkout; source truth and cross-dataset generalization unverified |
| Librosa preprocessing and acoustic features | Shared 16 kHz mono decoding and feature extraction used for training/inference | WAV/MP3 uploads; FLAC offline only; 20 MiB/60 s limits; waveform may preview a longer local file before server rejection |
| CNN/Transformer baseline | Three-feature standardized logistic regression | CNN/Transformer baseline is **not implemented**; logistic regression is a smaller experimental substitute |
| Hugging Face AST and PyTorch fine-tuning | No such training/inference path | Requires suitable dataset, compute, reproducible experiments and independent evaluation; not safe to substitute invented results |
| RIR/environmental reverberation investigation | Separate isolated-RIR endpoint: T20-derived RT60, C50, approximate DRR, synthetic decay regression tests | No RIR reconstruction from speech and no RIR-based deepfake detector; analyzer assumes caller supplies an actual RIR, cannot authenticate input type; operates after 16 kHz resampling |
| Breathing/cadence/syllable alignment | Spectral flatness, RMS variation and positive spectral flux only | These are **not** breathing, cadence or syllable-alignment measurements; no such validated subsystem |
| Inference API: label, confidence, version and time | Both variant interfaces; real/fake class scores; SHA-256 artifact identity and server processing time | Scores are uncalibrated; time includes decoding/model loading/upload reads, excludes client transit; no forensic verdict or latency SLA |
| Suspicious segment evidence | No fabricated segments returned or rendered | Global-feature classifiers do not localize manipulation; segment markers unavailable |
| Upload, waveform, states and retry | WAV/MP3 selection, validation, WaveSurfer playback/seek/remove; request cancellation, timeouts and retry | Inspection and classifier have separate file selectors; select the same clip when comparing; browser codec support varies |
| History | Latest 20 successful predictions kept in React memory; clear control | Page-session only; reload/tab close clears it; no audio retention, database or cross-device history |
| Responsive/accessibility testing | Semantic controls, labels and responsive CSS; browser regression suite | Automated Chromium checks cover selected flows, not a full accessibility or cross-browser certification |
| Docker and CI/CD | Multi-stage single-origin frontend/backend image; non-root process; read-only model mount; healthcheck; backend/frontend/browser/container CI jobs | No public deployment, TLS, authentication, global request-body/concurrency quotas, rate limits or production observability. Local research use only; CI is validation, not automatic production delivery |
| Architecture, API, evaluation, limitations and demo | README, API contract, this audit, verification report, existing dataset/experiment guides | Original numerical results remain historical self-reported claims until source artifacts/data can be rerun |

## Research interpretation

The three-feature baseline uses RMS amplitude, zero-crossing rate and spectral centroid. The six-feature candidate adds spectral flatness, RMS variation and positive spectral flux. Both are standardized logistic regressions. Neither isolates room acoustics from speaker, linguistic content, codec, microphone or synthesis artifacts.

The RIR analyzer consumes a **separate measured/simulated impulse-response WAV**. It has no link that converts its room-decay metrics into a speech-authenticity prediction. A plausible decay measurement does not prove a recording genuine. Near-ideal synthetic decay tests verify numerical behavior, not real-room validity or deepfake accuracy.

Reported historical baseline/candidate numbers must not be relabeled AST, CNN, RIR, independent benchmark results or deployment accuracy. The historical held-out split was drawn from official development speakers, not the official ASVspoof evaluation partition. Repeated selection against that split can bias its estimates. The comparison endpoint checks arithmetic consistency, not report provenance or linkage to the installed model hash.

## Reproduced bugs fixed in this audit

1. A trusted-but-malformed artifact containing `features=None` raised `TypeError` from `/api/model/status`. Validate metadata container types before conversion; reject fitted input widths that disagree with declared features.
2. A report with an entirely wrong confusion matrix could claim perfect scores and still be displayed. Both report readers now recompute accuracy, fake precision/recall/F1 from the matrix and reject contradictions (absolute tolerance 0.0001 for rounded reports).
3. Status/report fetches could wait indefinitely. Both now time out at 15 s and offer retry; cleanup prevents stale requests from changing current state.
4. Malformed optional evaluation metadata could crash the prediction panel. Validate the full status/evaluation shape before rendering and show a recoverable error instead.
5. CPU-bound decode/feature/inference calls in async upload handlers occupied the event loop. They now run in the server thread pool. This improves responsiveness, but does not establish a concurrency capacity guarantee.

## Required work before any research-complete claim

Provide the original trusted model files, their dependency versions, manifests, source recordings and evaluation JSON. Run the release preflight with a real clip and independently rerun held-out evaluations without using those results to tune. For the original RIR/AST objective, define and validate a speech-to-room-feature method, implement neural baselines and AST experiments, and report failure cases with suitable controls. Breathing/alignment and segment localization require separate experimental evidence. These cannot be responsibly completed by relabeling the present features. There is also no voice-activity gate, out-of-distribution detector, abstention policy or calibrated confidence; silence, music and unsupported recording conditions can still receive meaningless class scores.
