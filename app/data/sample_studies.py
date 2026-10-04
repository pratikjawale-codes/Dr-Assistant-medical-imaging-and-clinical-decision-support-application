"""
Synthetic Medical Study Factory & Clinical Pre-seeder
Generates realistic thoracic and musculoskeletal radiographs for out-of-the-box demonstration.
"""

from typing import Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..dicom.windowing import WindowLevelEngine
from ..models.inference import ModelInferenceEngine
from ..workflow.worklist import WorklistManager


class SampleStudiesFactory:
    """Creates synthetic radiographs and seeds the initial radiologist worklist."""

    @classmethod
    def generate_chest_xray_array(cls, pathology: str = "pneumonia", width: int = 512, height: int = 512) -> np.ndarray:
        """
        Synthesizes an anatomically structured Chest Radiograph:
        Thoracic cavity, bilateral lung fields, cardiac silhouette, ribs, and pathology opacities.
        """
        img = Image.new("L", (width, height), 25)
        draw = ImageDraw.Draw(img)

        # 1. Soft tissue outer contour & thoracic wall
        draw.ellipse([width * 0.08, height * 0.12, width * 0.92, height * 0.92], fill=45)

        # 2. Bilateral dark lung fields (radiolucent air spaces)
        # Left lung (anatomical right side of image)
        draw.ellipse([width * 0.14, height * 0.22, width * 0.44, height * 0.85], fill=190)
        # Right lung (anatomical left side of image)
        draw.ellipse([width * 0.56, height * 0.22, width * 0.86, height * 0.85], fill=190)

        # 3. Rib shadows (horizontal radiopaque arches)
        for y_offset in range(int(height * 0.26), int(height * 0.80), int(height * 0.07)):
            draw.arc([width * 0.12, y_offset, width * 0.46, y_offset + int(height * 0.12)], start=160, end=340, fill=85, width=9)
            draw.arc([width * 0.54, y_offset, width * 0.88, y_offset + int(height * 0.12)], start=200, end=20, fill=85, width=9)

        # 4. Central mediastinum & spine column
        draw.rectangle([width * 0.44, height * 0.10, width * 0.56, height * 0.95], fill=65)

        # 5. Cardiac Silhouette (radiopaque soft tissue)
        if pathology == "cardiomegaly":
            # Greatly enlarged cardiac shadow encroaching into left hemithorax
            draw.ellipse([width * 0.34, height * 0.46, width * 0.76, height * 0.86], fill=60)
        else:
            # Normal cardiothoracic ratio (< 0.50)
            draw.ellipse([width * 0.38, height * 0.50, width * 0.64, height * 0.82], fill=60)

        # 6. Specific Pathology Opacities
        if pathology == "pneumonia":
            # Right lower lobe dense alveolar consolidation with ill-defined borders
            for _ in range(8):
                cx = int(width * 0.28 + np.random.randint(-15, 15))
                cy = int(height * 0.62 + np.random.randint(-15, 15))
                rad = np.random.randint(30, 55)
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=95)
        elif pathology == "pneumothorax":
            # Right upper peripheral zone: hyperlucent absence of lung markings and pleural line
            draw.line([width * 0.78, height * 0.25, width * 0.82, height * 0.58], fill=40, width=3)
            draw.rectangle([width * 0.80, height * 0.22, width * 0.88, height * 0.65], fill=230)

        # Smooth image with Gaussian blur for organic medical radiograph realism
        img = img.filter(ImageFilter.GaussianBlur(radius=3.5))
        arr = np.array(img, dtype=np.float32)

        # Add simulated Poisson / quantum mottle quantum sensor noise
        noise = np.random.normal(0, 4.2, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255)

        return arr

    @classmethod
    def generate_wrist_fracture_array(cls, width: int = 512, height: int = 512) -> np.ndarray:
        """Synthesizes a distal radius extremity radiograph with cortical disruption fracture."""
        img = Image.new("L", (width, height), 20)
        draw = ImageDraw.Draw(img)

        # Forearm soft tissue silhouette
        draw.polygon([(width * 0.25, height * 0.95), (width * 0.30, height * 0.05),
                      (width * 0.70, height * 0.05), (width * 0.75, height * 0.95)], fill=45)

        # Radius bone (lateral)
        draw.polygon([(width * 0.33, height * 0.95), (width * 0.35, height * 0.25),
                      (width * 0.52, height * 0.22), (width * 0.48, height * 0.95)], fill=180)

        # Ulna bone (medial)
        draw.polygon([(width * 0.54, height * 0.95), (width * 0.55, height * 0.26),
                      (width * 0.66, height * 0.24), (width * 0.64, height * 0.95)], fill=175)

        # Cortical fracture disruption step-off
        draw.line([width * 0.34, height * 0.48, width * 0.51, height * 0.49], fill=40, width=4)

        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 3.5, arr.shape).astype(np.float32)
        return np.clip(arr + noise, 0, 255)

    @classmethod
    def seed_initial_studies(cls) -> List[Dict[str, Any]]:
        """Pre-seeds the platform with diverse clinical studies ready for immediate review."""
        seed_configs = [
            {
                "id": "ST-CXR-8401",
                "patient_name": "ANON^PATIENT^62M",
                "patient_id": "PT-94021",
                "patient_age": "062Y",
                "patient_sex": "M",
                "modality": "CR",
                "body_part": "CHEST",
                "pathology": "pneumonia",
                "clinical_indication": "Fever (39.1C), productive purulent cough, right pleuritic chest pain for 3 days."
            },
            {
                "id": "ST-CXR-8402",
                "patient_name": "ANON^PATIENT^71F",
                "patient_id": "PT-31950",
                "patient_age": "071Y",
                "patient_sex": "F",
                "modality": "CR",
                "body_part": "CHEST",
                "pathology": "cardiomegaly",
                "clinical_indication": "Exertional orthopnea, bilateral lower extremity edema, history of congestive heart failure."
            },
            {
                "id": "ST-CXR-8403",
                "patient_name": "ANON^PATIENT^24M",
                "patient_id": "PT-77142",
                "patient_age": "024Y",
                "patient_sex": "M",
                "modality": "CR",
                "body_part": "CHEST",
                "pathology": "pneumothorax",
                "clinical_indication": "Sudden onset sharp right chest pain after coughing fit, acute dyspnea, tachycardia."
            },
            {
                "id": "ST-MSK-8404",
                "patient_name": "ANON^PATIENT^35F",
                "patient_id": "PT-60293",
                "patient_age": "035Y",
                "patient_sex": "F",
                "modality": "DX",
                "body_part": "EXTREMITY_WRIST",
                "pathology": "fracture",
                "clinical_indication": "Fall on outstretched hand (FOOSH), acute tenderness over distal radial styloid."
            },
            {
                "id": "ST-CXR-8405",
                "patient_name": "ANON^PATIENT^45M",
                "patient_id": "PT-11204",
                "patient_age": "045Y",
                "patient_sex": "M",
                "modality": "CR",
                "body_part": "CHEST",
                "pathology": "normal",
                "clinical_indication": "Routine pre-operative clearance examination for elective laparoscopic cholecystectomy."
            }
        ]

        seeded_studies = []

        for cfg in seed_configs:
            # Generate pixel matrix
            if cfg["pathology"] == "fracture":
                pixels = cls.generate_wrist_fracture_array()
            else:
                pixels = cls.generate_chest_xray_array(pathology=cfg["pathology"])

            metadata = {
                "study_instance_uid": f"1.2.840.113619.2.55.3.{cfg['id'].replace('-', '')}",
                "patient_id": cfg["patient_id"],
                "patient_name": cfg["patient_name"],
                "patient_age": cfg["patient_age"],
                "patient_sex": cfg["patient_sex"],
                "modality": cfg["modality"],
                "body_part_examined": cfg["body_part"],
                "filename": f"{cfg['id']}_{cfg['pathology']}.dcm",
                "clinical_indication": cfg["clinical_indication"]
            }

            # Run inference pipeline
            ai_result = ModelInferenceEngine.run_inference(pixels, model_id="efficientnet_b0_cxr", metadata=metadata)
            
            # Base preview thumbnail
            windowed = WindowLevelEngine.apply_window_level(pixels, center=128, width=256)
            base_png = WindowLevelEngine.array_to_base64_png(windowed)

            study_record = {
                "study_id": cfg["id"],
                "patient_info": metadata,
                "modality": cfg["modality"],
                "body_part": cfg["body_part"],
                "clinical_indication": cfg["clinical_indication"],
                "status": "PENDING_REVIEW",
                "image_data_base64": base_png,
                "dimensions": [512, 512],
                "triage": ai_result["triage"],
                "ai_result": ai_result,
                "raw_pixels": pixels  # In-memory store for real-time windowing
            }

            WorklistManager.register_study(cfg["id"], study_record)
            seeded_studies.append(study_record)

        return seeded_studies
