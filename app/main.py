"""
AI-Powered Medical Imaging & Clinical Decision Support Platform
FastAPI Application Entry Point & DICOMweb Service
"""

import uuid
from typing import Dict, Any, List, Optional, Literal
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
import numpy as np

from .config import APP_NAME, APP_VERSION, REGULATORY_DISCLAIMER, STATIC_DIR, MODELS_CATALOG, WINDOW_PRESETS
from .dicom.parser import DICOMParser
from .dicom.anonymizer import DICOMAnonymizer
from .dicom.windowing import WindowLevelEngine
from .models.inference import ModelInferenceEngine
from .workflow.worklist import WorklistManager
from .workflow.feedback import ClinicalFeedbackStore
from .workflow.monitoring import MLOpsMonitor
from .data.quality import DataQualityAuditor
from .data.sample_studies import SampleStudiesFactory

# Initialize FastAPI App
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="End-to-End Medical Imaging AI/ML Engineering Platform Prototype"
)

# In-memory storage for raw pixel matrices (for fast window/level recalibration)
STUDY_RAW_PIXELS: Dict[str, np.ndarray] = {}

# Startup Event: Seed synthetic clinical cohorts
@app.on_event("startup")
def startup_event():
    print(f"[*] Initializing {APP_NAME} v{APP_VERSION}...")
    seeded = SampleStudiesFactory.seed_initial_studies()
    for s in seeded:
        STUDY_RAW_PIXELS[s["study_id"]] = s["raw_pixels"]
    print(f"[+] Platform ready. Loaded {len(seeded)} priority clinical studies into triage queue.")


# ==========================================
# 1. API Design Endpoints (per Plan Spec)
# ==========================================

@app.get("/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint providing service & hardware telemetry."""
    sys_metrics = MLOpsMonitor.get_system_metrics()
    return {
        "status": "HEALTHY",
        "platform": APP_NAME,
        "version": APP_VERSION,
        "regulatory_disclaimer": REGULATORY_DISCLAIMER,
        "telemetry": sys_metrics
    }


@app.get("/models")
def get_models() -> List[Dict[str, Any]]:
    """Returns catalog of registered medical AI models (ResNet, EfficientNet, MobileNet, U-Net)."""
    return MLOpsMonitor.get_model_registry()


@app.get("/models/{model_id}")
def get_model_detail(model_id: str) -> Dict[str, Any]:
    """Returns detailed architecture & evaluation metrics for a specific model."""
    if model_id not in MODELS_CATALOG:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")
    return MODELS_CATALOG[model_id]


@app.get("/studies")
def list_studies(
    priority: Optional[str] = Query(None, description="Filter by triage priority (STAT, URGENT, ROUTINE, ALL)"),
    status: Optional[str] = Query(None, description="Filter by review status (PENDING_REVIEW, SIGNED_OFF, OVERRIDDEN, ALL)")
) -> List[Dict[str, Any]]:
    """Lists prioritized radiologist worklist queue sorted by urgency."""
    studies = WorklistManager.list_worklist(priority_filter=priority, status_filter=status)
    # Strip raw pixel arrays to keep JSON response lightweight
    clean_list = []
    for s in studies:
        item = dict(s)
        item.pop("raw_pixels", None)
        item.pop("image_data_base64", None)
        ai_result = item.get("ai_result")
        if ai_result:
            ai_result = dict(ai_result)
            explainability = dict(ai_result.get("explainability", {}))
            explainability.pop("gradcam_overlay_base64", None)
            explainability.pop("bbox_overlay_base64", None)
            ai_result["explainability"] = explainability
            segmentation = dict(ai_result.get("segmentation", {}))
            segmentation.pop("mask_overlay_base64", None)
            ai_result["segmentation"] = segmentation
            item["ai_result"] = ai_result
        clean_list.append(item)
    return clean_list


@app.get("/studies/{study_id}")
def get_study(study_id: str) -> Dict[str, Any]:
    """Retrieves full metadata and image preview for a specific study."""
    study = WorklistManager.get_study(study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found.")
    res = dict(study)
    res.pop("raw_pixels", None)
    return res


@app.get("/studies/{study_id}/result")
def get_study_result(study_id: str) -> Dict[str, Any]:
    """Returns AI inference results, Grad-CAM overlays, segmentation masks, and triage signals."""
    study = WorklistManager.get_study(study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found.")
    return {
        "study_id": study_id,
        "triage": study.get("triage"),
        "ai_result": study.get("ai_result"),
        "review_record": study.get("review_record")
    }


# Request Model for Radiologist Review Feedback
class FeedbackPayload(BaseModel):
    study_id: str
    action: Literal["ACCEPT", "MODIFY", "REJECT"]
    final_diagnosis: str = Field(min_length=1)
    clinical_impressions: str
    radiologist_id: Optional[str] = "Dr. S. Chen, MD (Radiology)"
    override_rationale: Optional[str] = None


@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload) -> Dict[str, Any]:
    """
    Submits radiologist review (Accept/Modify/Reject), signs off the study,
    generates DICOM Structured Report (SR), and commits record to audit trail.
    """
    study = WorklistManager.get_study(payload.study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Study '{payload.study_id}' not found.")

    predictions = study.get("ai_result", {}).get("predictions", [])
    ai_finding = predictions[0].get("finding", "Unknown") if predictions else "Unknown"
    ai_conf = predictions[0].get("probability", 0.0) if predictions else 0.0

    # Record in feedback store
    record = ClinicalFeedbackStore.record_feedback(
        study_id=payload.study_id,
        radiologist_id=payload.radiologist_id,
        action=payload.action,
        final_diagnosis=payload.final_diagnosis,
        impressions=payload.clinical_impressions,
        ai_finding=ai_finding,
        ai_confidence=ai_conf,
        override_rationale=payload.override_rationale
    )

    # Update study status
    new_status = "SIGNED_OFF" if payload.action.upper() == "ACCEPT" else "OVERRIDDEN"
    WorklistManager.update_status(payload.study_id, new_status, radiologist=payload.radiologist_id)
    study["review_record"] = record
    study["status"] = new_status

    return {
        "status": "SUCCESS",
        "message": f"Study '{payload.study_id}' signed off with status: {new_status}",
        "feedback_record": record
    }


# DICOM / Image Upload & Ingestion Endpoint
@app.post("/studies")
async def upload_study(
    file: UploadFile = File(...),
    model_id: str = Form("efficientnet_b0_cxr"),
    clinical_indication: str = Form("Clinical radiograph submitted for triage assessment.")
) -> Dict[str, Any]:
    """
    Ingests DICOM or raster scan, parses metadata, performs HIPAA de-identification,
    executes AI inference & explainability, and registers study into radiologist worklist.
    """
    contents = await file.read()
    parse_result = DICOMParser.parse_file(contents, file.filename or "upload")
    
    if not parse_result.get("success"):
        raise HTTPException(status_code=400, detail=parse_result.get("error", "Failed to parse image"))

    raw_metadata = parse_result["metadata"]
    pixel_array = parse_result["pixel_array"]

    # Automated HIPAA de-identification
    anon_metadata = DICOMAnonymizer.anonymize_metadata(raw_metadata)

    # Run AI inference
    if model_id not in MODELS_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'.")
    ai_result = ModelInferenceEngine.run_inference(pixel_array, model_id=model_id, metadata=anon_metadata)

    # Base display windowing
    windowed = WindowLevelEngine.apply_window_level(
        pixel_array,
        center=anon_metadata.get("window_center", 128),
        width=anon_metadata.get("window_width", 256),
        invert=anon_metadata.get("photometric_interpretation") == "MONOCHROME1",
    )
    image_b64 = WindowLevelEngine.array_to_base64_png(windowed)

    # Generate Unique Study ID
    study_id = f"ST-UPL-{uuid.uuid4().hex[:10].upper()}"
    anon_metadata["clinical_indication"] = clinical_indication

    study_record = {
        "study_id": study_id,
        "patient_info": anon_metadata,
        "modality": anon_metadata.get("modality", "CR"),
        "body_part": anon_metadata.get("body_part_examined", "CHEST"),
        "clinical_indication": clinical_indication,
        "status": "PENDING_REVIEW",
        "image_data_base64": image_b64,
        "dimensions": list(pixel_array.shape[:2]),
        "triage": ai_result["triage"],
        "ai_result": ai_result,
        "raw_pixels": pixel_array
    }

    WorklistManager.register_study(study_id, study_record)
    STUDY_RAW_PIXELS[study_id] = pixel_array

    # Clean response
    resp = dict(study_record)
    resp.pop("raw_pixels", None)
    return resp


class InferenceRequest(BaseModel):
    study_id: Optional[str] = None
    model_id: Optional[str] = "efficientnet_b0_cxr"


# Dedicated Inference Endpoint (Plan Section 7)
@app.post("/inference")
async def run_standalone_inference(request: Request) -> Dict[str, Any]:
    """
    Direct model inference endpoint per API Design spec.
    Can run inference on an existing study_id or on an uploaded file stream.
    """
    payload: Optional[InferenceRequest] = None
    file: Optional[UploadFile] = None
    chosen_model = "efficientnet_b0_cxr"

    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("application/json"):
        try:
            payload = InferenceRequest.model_validate(await request.json())
        except (ValueError, TypeError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON inference request.") from exc
        chosen_model = payload.model_id or chosen_model
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploaded_file = form.get("file")
        if isinstance(uploaded_file, StarletteUploadFile):
            file = uploaded_file
        study_id = form.get("study_id")
        chosen_model = str(form.get("model_id") or chosen_model)
        if study_id:
            payload = InferenceRequest(study_id=str(study_id), model_id=chosen_model)
    else:
        raise HTTPException(
            status_code=415,
            detail="Use application/json with study_id or multipart/form-data with a file.",
        )

    if chosen_model not in MODELS_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown model '{chosen_model}'.")

    if payload and payload.study_id:
        study = WorklistManager.get_study(payload.study_id)
        if not study:
            raise HTTPException(status_code=404, detail=f"Study '{payload.study_id}' not found.")
        pixels = STUDY_RAW_PIXELS.get(payload.study_id)
        if pixels is None:
            raise HTTPException(status_code=404, detail="Pixel array not found for study.")
        meta = study.get("patient_info", {})
        return ModelInferenceEngine.run_inference(pixels, model_id=chosen_model, metadata=meta)

    elif file:
        contents = await file.read()
        parse_result = DICOMParser.parse_file(contents, file.filename or "upload")
        if not parse_result.get("success"):
            raise HTTPException(status_code=400, detail="Invalid DICOM or image file.")
        return ModelInferenceEngine.run_inference(parse_result["pixel_array"], model_id=chosen_model, metadata=parse_result["metadata"])

    else:
        raise HTTPException(status_code=400, detail="Must provide either a JSON payload with study_id or a multipart file.")


# Window / Level Recalibration Endpoint
class WindowPayload(BaseModel):
    preset: Optional[str] = "default"
    center: Optional[float] = 128.0
    width: Optional[float] = 256.0
    invert: Optional[bool] = False


@app.post("/studies/{study_id}/window")
def adjust_window(study_id: str, payload: WindowPayload) -> Dict[str, Any]:
    """Applies real-time clinical window/level LUT adjustment to a study's pixel matrix."""
    if study_id not in STUDY_RAW_PIXELS:
        raise HTTPException(status_code=404, detail="Study pixel array not available.")

    pixels = STUDY_RAW_PIXELS[study_id]
    
    if payload.preset and payload.preset.lower() in WINDOW_PRESETS:
        windowed, preset_info = WindowLevelEngine.apply_preset(pixels, payload.preset, invert=payload.invert)
        center_val = preset_info["center"]
        width_val = preset_info["width"]
    else:
        center_val = float(payload.center or 128.0)
        width_val = float(payload.width or 256.0)
        windowed = WindowLevelEngine.apply_window_level(pixels, center_val, width_val, invert=payload.invert)

    image_b64 = WindowLevelEngine.array_to_base64_png(windowed)
    return {
        "study_id": study_id,
        "window_center": center_val,
        "window_width": width_val,
        "invert": payload.invert,
        "image_base64": image_b64
    }


# MLOps Telemetry & Drift Endpoint
@app.get("/mlops/metrics")
def get_mlops_metrics() -> Dict[str, Any]:
    """Provides real-time drift metrics, population stability index, and feedback stats."""
    return {
        "telemetry": MLOpsMonitor.get_system_metrics(),
        "drift_analytics": MLOpsMonitor.get_drift_analysis(),
        "feedback_statistics": ClinicalFeedbackStore.get_feedback_statistics(),
        "models": MLOpsMonitor.get_model_registry(),
        "retraining_status": MLOpsMonitor.get_retraining_status()
    }


@app.post("/mlops/retrain")
def trigger_retraining() -> Dict[str, Any]:
    """Triggers feedback-driven active learning retraining pipeline."""
    return MLOpsMonitor.trigger_retraining()


@app.post("/mlops/promote")
def promote_model(model_id: str = Form(...)) -> Dict[str, Any]:
    """Promotes a model to production or executes rollback."""
    success = MLOpsMonitor.set_production_model(model_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to promote model.")
    return {"status": "SUCCESS", "active_production_model": model_id}


# Data Quality & Patient Leakage Audit Endpoint
class LeakageCheckPayload(BaseModel):
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


@app.post("/quality/leakage-check")
def check_patient_leakage(payload: LeakageCheckPayload) -> Dict[str, Any]:
    """Audits train/val/test cohorts for patient-level data leakage."""
    return DataQualityAuditor.check_patient_leakage(payload.train_ids, payload.val_ids, payload.test_ids)


@app.get("/quality/audit-study/{study_id}")
def audit_study_quality(study_id: str) -> Dict[str, Any]:
    """Performs image quality and missing-label checks on a loaded study."""
    study = WorklistManager.get_study(study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found.")
    pixels = STUDY_RAW_PIXELS.get(study_id)
    return DataQualityAuditor.audit_image(pixels, study.get("patient_info", {}))


# Mount Static Files and Serve Main UI
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the main single-page clinical workstation interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Dr.Assistant is starting up. Please refresh shortly.</h2>")
