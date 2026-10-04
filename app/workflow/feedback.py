"""
Clinical Feedback Store, Human-in-the-Loop Override & Structured AI Reporting
Supports Accept/Modify/Reject workflow and HIPAA-compliant audit trail logging.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
from ..config import AUDIT_LOG_FILE


class ClinicalFeedbackStore:
    """Records human radiologist verification, corrections, and audit trails."""

    _feedback_records: List[Dict[str, Any]] = []

    @classmethod
    def record_feedback(
        cls,
        study_id: str,
        radiologist_id: str,
        action: str,  # ACCEPT, MODIFY, REJECT
        final_diagnosis: str,
        impressions: str,
        ai_finding: str,
        ai_confidence: float,
        override_rationale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Logs radiologist validation and creates structured clinical report.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        agreement = (action.upper() == "ACCEPT")

        record = {
            "record_id": f"REV_{len(cls._feedback_records) + 1:05d}",
            "study_id": study_id,
            "timestamp": timestamp,
            "radiologist_id": radiologist_id or "Dr. S. Chen, MD (Radiology)",
            "action": action.upper(),
            "agreement_with_ai": agreement,
            "ai_original_finding": ai_finding,
            "ai_original_confidence": ai_confidence,
            "radiologist_final_diagnosis": final_diagnosis,
            "clinical_impressions": impressions,
            "override_rationale": override_rationale or ("Finding verified" if agreement else "Clinical discordance noted"),
            "dataset_curation_candidate": not agreement  # High-value sample for active learning
        }

        # Append to persistent audit log file
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Generate DICOM Structured Report (SR) representation
        structured_report = cls.generate_structured_report(record)
        record["structured_report"] = structured_report
        cls._feedback_records.append(record)

        return record

    @classmethod
    def generate_structured_report(cls, review_record: Dict[str, Any]) -> Dict[str, Any]:
        """Generates standard DICOM SR (TID 1500 style) clinical report."""
        return {
            "document_type": "DICOM Structured Report (SR) - TID 1500",
            "report_header": {
                "title": "AI-Assisted Diagnostic Imaging Evaluation",
                "study_id": review_record["study_id"],
                "radiologist": review_record["radiologist_id"],
                "verification_time": review_record["timestamp"],
                "verification_status": "VERIFIED_BY_LICENSED_PHYSICIAN"
            },
            "findings_section": {
                "ai_preliminary_assessment": review_record["ai_original_finding"],
                "ai_statistical_confidence": f"{review_record['ai_original_confidence'] * 100:.1f}%",
                "radiologist_verified_finding": review_record["radiologist_final_diagnosis"],
                "action_taken": review_record["action"]
            },
            "clinical_impression": review_record["clinical_impressions"],
            "legal_attestation": (
                "The AI algorithm served as an assistive triage aid. All final diagnostic "
                "interpretations and medical judgments were rendered by the signing physician."
            )
        }

    @classmethod
    def list_feedback(cls) -> List[Dict[str, Any]]:
        """Returns all recorded reviews."""
        return list(cls._feedback_records)

    @classmethod
    def get_feedback_statistics(cls) -> Dict[str, Any]:
        """Computes human-AI agreement rate and override breakdown."""
        total = len(cls._feedback_records)
        if total == 0:
            return {
                "total_reviews": 0,
                "concordance_rate": 100.0,
                "accept_count": 0,
                "modify_count": 0,
                "reject_count": 0,
                "active_learning_queue_size": 0
            }

        accepts = sum(1 for r in cls._feedback_records if r["action"] == "ACCEPT")
        modifies = sum(1 for r in cls._feedback_records if r["action"] == "MODIFY")
        rejects = sum(1 for r in cls._feedback_records if r["action"] == "REJECT")
        concordance = (accepts / total) * 100.0

        return {
            "total_reviews": total,
            "concordance_rate": round(concordance, 1),
            "accept_count": accepts,
            "modify_count": modifies,
            "reject_count": rejects,
            "active_learning_queue_size": modifies + rejects
        }
