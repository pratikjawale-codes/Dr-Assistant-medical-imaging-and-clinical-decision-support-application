# Dr.Assistant

A local medical-imaging workflow and clinical decision-support prototype. It provides a FastAPI service and a browser-based workstation for loading synthetic examples or local DICOM/raster images, reviewing illustrative model outputs, and recording a human review.

> **Important:** Dr.Assistant is a research/demo prototype, not a validated medical device. Its inference, segmentation, uncertainty, telemetry, and retraining behavior are illustrative simulations—not diagnostic AI. Do not use it to make patient-care decisions or upload real patient data.

## Requirements

- Python 3.11 or newer
- A browser
- Windows users can optionally launch with `run.bat`

## Run locally

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

Or, after activating your environment and installing requirements:

```powershell
python run.py
```

The launcher starts the local web server, waits until it accepts connections, and opens the app at <http://127.0.0.1:8000>. Use `Ctrl+C` in the terminal to stop it. The UI can be opened manually at that address if the browser does not open.

Optional local binding settings:

```powershell
$env:DR_ASSISTANT_HOST = "127.0.0.1"
$env:DR_ASSISTANT_PORT = "8000"
python run.py
```

The server binds to loopback by default. Use a non-loopback host only on a trusted network and with appropriate access controls.

## What the app does

1. Seeds five synthetic example studies when the server starts.
2. Lists studies in a priority-sorted worklist.
3. Opens a selected study with the image preview, overlays, triage metadata, and review form.
4. Accepts a DICOM file or common raster image upload and routes it through parsing, identifier masking, illustrative inference, and worklist registration.
5. Records Accept / Modify / Reject feedback, generates a structured-report-shaped JSON summary, and appends the review to `data_store/audit_trail.log`.
6. Exposes model, monitoring, and data-quality demo endpoints in the web tabs and REST API.

See [WORKFLOW.md](./WORKFLOW.md) for the end-to-end component and endpoint map.

## REST API overview

Interactive API docs are available at <http://127.0.0.1:8000/docs>.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status and demo telemetry |
| `GET` | `/studies` | Lightweight prioritized worklist; accepts `priority` and `status` filters |
| `POST` | `/studies` | Upload a DICOM or raster study; multipart fields: `file`, optional `model_id`, optional `clinical_indication` |
| `GET` | `/studies/{study_id}` | Study metadata and viewer image/overlay payloads |
| `GET` | `/studies/{study_id}/result` | Inference and review result |
| `POST` | `/studies/{study_id}/window` | Recalculate image preview using a preset or center/width |
| `POST` | `/inference` | JSON `{ "study_id": "...", "model_id": "..." }` or multipart `file` inference |
| `POST` | `/feedback` | Record radiologist review |
| `GET` | `/models`, `/models/{model_id}` | Model catalog |
| `GET` | `/mlops/metrics` | Simulated monitoring and registry information |
| `POST` | `/mlops/retrain`, `/mlops/promote` | Simulated registry/retraining actions |
| `POST` | `/quality/leakage-check` | Check patient IDs across train/validation/test lists |
| `GET` | `/quality/audit-study/{study_id}` | Image quality summary for a loaded study |

## Implementation notes

- Study/worklist state and uploaded pixel data are held in memory and reset on restart. Only the audit log is persisted locally.
- DICOM parsing uses `pydicom`; raster images are handled with Pillow. Optional deep-learning packages are intentionally not required because this prototype does not load trained model weights.
- The model names and scores describe sample/demo registry entries; no actual EfficientNet, ResNet, MobileNet, or MONAI model is executed.
- The audit log can contain user-entered review text. Treat it as sensitive, keep this demo on a trusted machine, and clear the generated `data_store` when appropriate.
- `docker-compose.yml` includes an optional Orthanc service for local experimentation; this app currently does not query or integrate with Orthanc.

## Project layout

```text
app/
  main.py                 FastAPI routes and startup seeding
  config.py               Settings, model catalog, and runtime data paths
  data/                   Synthetic cases and quality checks
  dicom/                  DICOM/raster parsing, anonymization, windowing
  models/                 Illustrative inference, overlays, segmentation, triage
  workflow/               Worklist, feedback/audit log, demo monitoring
  static/                 HTML, CSS, and browser client
run.py                    Local server launcher and browser opener
run.bat                   Windows launcher
requirements.txt          Runtime dependencies
WORKFLOW.md               Detailed flow and module explanations
```

