"""
AI-Powered Medical Imaging & Clinical Decision Support Platform
Application Configuration & Clinical Parameters
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
DATA_DIR = BASE_DIR / "data_store"
UPLOAD_DIR = DATA_DIR / "uploads"
ANONYMIZED_DIR = DATA_DIR / "anonymized"
AUDIT_LOG_FILE = DATA_DIR / "audit_trail.log"

for d in [DATA_DIR, UPLOAD_DIR, ANONYMIZED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Application Metadata
APP_NAME = "Dr.Assistant - Medical Imaging & Clinical Decision Support"
APP_VERSION = "1.0.0"
REGULATORY_DISCLAIMER = (
    "Research / portfolio prototype — not for autonomous clinical diagnosis or patient treatment. "
    "Designed for assistive clinical decision-support and radiologist triage workflow."
)

# Supported AI Models for Comparative Evaluation
MODELS_CATALOG = {
    "custom_baseline_cnn": {
        "id": "custom_baseline_cnn",
        "name": "Custom 4-Block Baseline CNN",
        "framework": "TensorFlow / Keras / PyTorch",
        "version": "v1.0",
        "status": "Benchmark",
        "auc_roc": 0.821,
        "f1_score": 0.785,
        "latency_ms": 18,
        "input_resolution": "256x256",
        "description": "Foundational 4-stage convolutional baseline (Conv2D -> BatchNorm -> MaxPool -> Dropout) for transfer learning ablation."
    },
    "efficientnet_b0_cxr": {
        "id": "efficientnet_b0_cxr",
        "name": "EfficientNet-B0 (Chest X-Ray Multi-label)",
        "framework": "PyTorch / MONAI",
        "version": "v2.1",
        "status": "Production",
        "auc_roc": 0.924,
        "f1_score": 0.887,
        "latency_ms": 32,
        "input_resolution": "512x512",
        "description": "Optimized compound scaling network trained on multi-cohort chest radiographs."
    },
    "resnet50_cxr": {
        "id": "resnet50_cxr",
        "name": "ResNet-50 Deep Residual Classifier",
        "framework": "PyTorch / MONAI",
        "version": "v1.8",
        "status": "Staging",
        "auc_roc": 0.908,
        "f1_score": 0.865,
        "latency_ms": 48,
        "input_resolution": "512x512",
        "description": "Standard deep residual benchmark for pulmonary pathology identification."
    },
    "mobilenet_v3_cxr": {
        "id": "mobilenet_v3_cxr",
        "name": "MobileNet-V3 (Edge & Point-of-Care)",
        "framework": "TensorFlow / Keras",
        "version": "v1.4",
        "status": "Staging",
        "auc_roc": 0.879,
        "f1_score": 0.832,
        "latency_ms": 14,
        "input_resolution": "256x256",
        "description": "Ultra-low latency model engineered for mobile triage and rural health clinics."
    },
    "monai_unet_lesion": {
        "id": "monai_unet_lesion",
        "name": "MONAI 2D/3D U-Net Segmentation",
        "framework": "MONAI + PyTorch",
        "version": "v3.0",
        "status": "Production",
        "dice_score": 0.862,
        "iou_score": 0.768,
        "latency_ms": 65,
        "input_resolution": "512x512",
        "description": "Volumetric & planar segmentation of pulmonary infiltrates, nodules and consolidation."
    }
}

# Windowing Presets for Medical CT / Radiography
WINDOW_PRESETS = {
    "default": {"name": "Standard Radiograph", "center": 128, "width": 256},
    "lung": {"name": "Lung Window", "center": -600, "width": 1500},
    "bone": {"name": "Bone Window", "center": 300, "width": 2000},
    "soft_tissue": {"name": "Soft Tissue / Mediastinum", "center": 40, "width": 350},
    "brain": {"name": "Brain CT Window", "center": 40, "width": 80}
}

# Triage & Urgency Rules
TRIAGE_THRESHOLDS = {
    "STAT": 0.78,    # Critical findings (Pneumothorax, massive effusion, severe consolidation)
    "URGENT": 0.45,  # Moderate findings (Infiltrate, fracture, mild cardiomegaly)
    "ROUTINE": 0.00  # Normal or minor non-actionable changes
}

# HIPAA Safe Harbor Tag Identifiers for De-identification
HIPAA_PHI_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "InstitutionName", "ReferringPhysicianName", "AccessionNumber",
    "StudyID", "StationName", "InstitutionalDepartmentName"
]
