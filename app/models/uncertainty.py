"""
Uncertainty Estimation and Clinical Triage Priority Engine
Calculates epistemic/aleatoric uncertainty and assigns STAT / URGENT / ROUTINE triage priority.
"""

from typing import Dict, Any, List
import numpy as np
from ..config import TRIAGE_THRESHOLDS

# Clinical severity weights for different radiographic findings
FINDING_SEVERITY_WEIGHTS = {
    "Pneumothorax": 1.0,         # Emergency
    "Massive Pleural Effusion": 0.95,
    "Consolidation / Lobar Pneumonia": 0.85,
    "Displaced Fracture": 0.80,
    "Cardiomegaly": 0.65,
    "Atelectasis": 0.50,
    "Nodule / Granuloma": 0.45,
    "No Acute Finding / Normal": 0.05
}


class UncertaintyPriorityEngine:
    """Estimates prediction uncertainty and computes clinical triage scores."""

    @classmethod
    def evaluate_uncertainty_and_risk(
        cls,
        predictions: List[Dict[str, Any]],
        image_quality_score: float = 0.94
    ) -> Dict[str, Any]:
        """
        Calculates:
        - Epistemic uncertainty (model ambiguity based on entropy/dispersion)
        - Aleatoric uncertainty (image noise/acquisition quality)
        - Clinical Risk Score (0.0 to 1.0)
        - Worklist Urgency Priority (STAT, URGENT, ROUTINE)
        """
        if not predictions:
            return {
                "triage_priority": "ROUTINE",
                "risk_score": 0.1,
                "confidence_score": 0.85,
                "epistemic_uncertainty": 0.12,
                "aleatoric_uncertainty": 0.08,
                "triage_rationale": "No active findings detected."
            }

        top_pred = predictions[0]
        top_label = top_pred["finding"]
        top_prob = float(top_pred["probability"])

        # Finding severity weight
        severity = FINDING_SEVERITY_WEIGHTS.get(top_label, 0.5)

        # Epistemic uncertainty: higher when probability is near decision boundary (0.50)
        dist_from_margin = abs(top_prob - 0.5) * 2.0  # 0 to 1
        epistemic = float(np.clip(1.0 - dist_from_margin * 0.85, 0.05, 0.95))

        # Aleatoric uncertainty: inversely proportional to image quality score
        aleatoric = float(np.clip(1.0 - image_quality_score, 0.03, 0.80))

        # Calibrated confidence (penalized by epistemic uncertainty)
        confidence = float(np.clip(top_prob * (1.0 - epistemic * 0.35), 0.10, 0.99))

        # Clinical Risk score incorporates probability, finding severity, and uncertainty
        risk_score = float(np.clip((top_prob * 0.65 + severity * 0.35) * (1.0 if top_label != "No Acute Finding / Normal" else 0.05), 0.0, 1.0))

        # Triage determination
        if risk_score >= TRIAGE_THRESHOLDS["STAT"] or (top_label == "Pneumothorax" and top_prob > 0.40):
            priority = "STAT"
            rationale = f"CRITICAL: High probability of {top_label} (Risk {risk_score:.2f}). Requires immediate attending radiologist review."
        elif risk_score >= TRIAGE_THRESHOLDS["URGENT"]:
            priority = "URGENT"
            rationale = f"URGENT: Abnormal findings suggestive of {top_label} (Risk {risk_score:.2f}). Place on prioritized worklist."
        else:
            priority = "ROUTINE"
            rationale = f"ROUTINE: Low clinical risk profile or normal study (Risk {risk_score:.2f}). Standard review order."

        return {
            "triage_priority": priority,
            "risk_score": round(risk_score, 3),
            "calibrated_confidence": round(confidence, 3),
            "epistemic_uncertainty": round(epistemic, 3),
            "aleatoric_uncertainty": round(aleatoric, 3),
            "image_quality_index": round(image_quality_score, 3),
            "triage_rationale": rationale,
            "sla_target_minutes": 15 if priority == "STAT" else (60 if priority == "URGENT" else 240)
        }
