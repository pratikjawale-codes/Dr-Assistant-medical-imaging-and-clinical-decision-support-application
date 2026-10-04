# Dr.Assistant: project workflow and component guide

## 1. Start-up and request flow

Run `python run.py` from the repository root (or use `run.bat` on Windows):

1. `run.py` validates the configured port and starts a readiness watcher.
2. Uvicorn imports `app.main:app` and starts the FastAPI application.
3. The startup handler in `app/main.py` calls `SampleStudiesFactory.seed_initial_studies()`.
4. The sample factory creates five synthetic images, runs the illustrative inference pipeline, builds PNG previews, and registers each case with `WorklistManager`.
5. Raw pixel arrays are kept in the process-local `STUDY_RAW_PIXELS` map for window/level changes, inference requests, and quality checks.
6. Once the server is accepting connections, `run.py` opens the browser. The page loads its static assets and fetches the worklist/model catalog.

The default address is `http://127.0.0.1:8000`. Set `DR_ASSISTANT_HOST` and `DR_ASSISTANT_PORT` to change the listener. In-memory studies disappear when the process stops.

## 2. Source map

| File or package | Responsibility |
| --- | --- |
| `run.py` | Local Uvicorn launcher, environment-based bind settings, readiness check, browser launch |
| `app/main.py` | FastAPI setup, startup seeding, HTTP routes, in-memory pixel cache, frontend serving |
| `app/config.py` | Application metadata, paths, sample model catalog, window presets, triage thresholds |
| `app/data/sample_studies.py` | Synthetic X-ray creation and worklist bootstrapping |
| `app/data/quality.py` | Pixel-quality summary and patient split overlap detection |
| `app/dicom/parser.py` | DICOM parsing through pydicom and raster decoding through Pillow |
| `app/dicom/anonymizer.py` | Prototype field masking and deterministic patient pseudonym generation |
| `app/dicom/windowing.py` | DICOM-style window/level conversion and PNG data URLs |
| `app/models/inference.py` | Demo finding selection and orchestration of illustrative outputs |
| `app/models/explainability.py` | Synthetic activation, bounding-box and overlay image generation |
| `app/models/segmentation.py` | Synthetic mask and approximate area/benchmark values |
| `app/models/uncertainty.py` | Demo risk/uncertainty values and STAT/URGENT/ROUTINE assignment |
| `app/workflow/worklist.py` | In-memory study registration, filtering, ordering, and review status |
| `app/workflow/feedback.py` | Review record, JSON-lines audit append, report-shaped response |
| `app/workflow/monitoring.py` | Demo model registry, telemetry, drift and retraining responses |
| `app/static/index.html` | Worklist, viewer, review, monitoring and data-quality page structure |
| `app/static/css/style.css` | Pearl palette, responsive layout refinements and reusable UI styling |
| `app/static/js/app.js` | Browser navigation, API calls, rendering, image controls and review actions |

## 3. Study ingestion and viewing

`POST /studies` accepts multipart form data:

1. `DICOMParser.parse_file()` decodes a DICOM image or supported raster image and returns pixels plus extracted/demo metadata.
2. `DICOMAnonymizer.anonymize_metadata()` replaces selected direct identifiers and caps ages above 89.
3. `ModelInferenceEngine.run_inference()` returns demo predictions, a triage payload, comparison rows, overlay images, and a segmentation-shaped result.
4. `WindowLevelEngine` renders the base image preview. DICOM center/width and MONOCHROME1 inversion are respected for the initial display.
5. A unique study ID is created; the study is registered in `WorklistManager` and the pixels are cached for this server process.
6. The endpoint returns the full study response so the browser can open it immediately.

`GET /studies` deliberately returns a lightweight worklist without image data URLs. The browser selects an item and requests `GET /studies/{study_id}` for its preview and overlays. This prevents transferring every full-size image when the worklist loads.

The viewer composes the base image, Grad-CAM-style heatmap, bounding box, and mask on a canvas. Window presets call `POST /studies/{study_id}/window`; pan, zoom, inversion and layer toggles are applied client-side.

## 4. Review and feedback

The viewer's Accept, Modify and Reject buttons send a JSON payload to `POST /feedback` containing `study_id`, `action`, `final_diagnosis`, `clinical_impressions`, `radiologist_id`, and an optional override rationale.

The API validates the action, finds the corresponding worklist study, derives the first demo finding, and asks `ClinicalFeedbackStore` to create a review record. The store appends that record to `data_store/audit_trail.log`, then returns a report-shaped JSON summary. The study status becomes `SIGNED_OFF` for Accept or `OVERRIDDEN` for Modify/Reject. The page shows the returned report and refreshes the worklist.

If the audit log cannot be written, the review request fails instead of reporting a successful saved review. This log is not a standards-certified DICOM Structured Report file; it is a JSON representation for this prototype.

## 5. API and browser crosswalk

| Browser action | API route(s) | Main implementation |
| --- | --- | --- |
| Initial page / health | `GET /`, `GET /health` | `app/main.py` and `app/static/index.html` |
| Refresh/filter worklist | `GET /studies` | `WorklistManager.list_worklist()` |
| Open a study | `GET /studies/{study_id}` | `WorklistManager.get_study()` |
| Upload a scan | `POST /studies` | Parser → identifier masking → illustrative inference → worklist |
| Adjust window | `POST /studies/{study_id}/window` | `WindowLevelEngine` and cached pixels |
| Sign off or override | `POST /feedback` | `ClinicalFeedbackStore` and `WorklistManager.update_status()` |
| Model/monitor tab | `GET /models`, `GET /mlops/metrics` | `MLOpsMonitor` |
| Trigger demo retraining | `POST /mlops/retrain` | `MLOpsMonitor.trigger_retraining()` |
| Promote demo model | `POST /mlops/promote` | `MLOpsMonitor.set_production_model()` |
| Audit patient split | `POST /quality/leakage-check` | `DataQualityAuditor.check_patient_leakage()` |
| Audit loaded image | `GET /quality/audit-study/{study_id}` | `DataQualityAuditor.audit_image()` |
| Standalone inference | `POST /inference` | JSON study ID or multipart file → inference engine |

## 6. Runtime data and limitations

- The study store, review list, model registry changes, and pixel cache live in process memory.
- The audit trail is appended as JSON Lines under `data_store/audit_trail.log`.
- Synthetic example images are generated at startup; they are not patient data.
- Finding labels, confidence values, baseline comparisons, masks, metrics, and retraining status are illustrative. No trained neural network weights are loaded and no DICOMweb/PACS exchange is implemented.
- Identifier masking is a demonstration, not a claim of regulatory compliance. DICOM attributes and embedded image pixels can retain sensitive information; never upload real patient data to this prototype.
- The UI and API do not replace clinical review or validated clinical software.

