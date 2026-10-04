"""
Radiologist Priority Worklist & Clinical Queue Manager
Simulates AI-assisted PACS triage worklist with SLA tracking and priority escalation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

class WorklistManager:
    """Manages clinical studies, priority queues, and radiologist worklists."""

    _studies: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_study(cls, study_id: str, study_data: Dict[str, Any]) -> Dict[str, Any]:
        """Registers or updates a study in the worklist store."""
        created_at = study_data.get("created_at", _utc_now().isoformat())
        priority = study_data.get("triage", {}).get("triage_priority", "ROUTINE")
        
        sla_minutes = 15 if priority == "STAT" else (60 if priority == "URGENT" else 240)
        due_at = (_utc_now() + timedelta(minutes=sla_minutes)).isoformat()

        record = {
            "study_id": study_id,
            "created_at": created_at,
            "due_at": due_at,
            "sla_target_minutes": sla_minutes,
            "status": study_data.get("status", "PENDING_REVIEW"),  # PENDING_REVIEW, IN_REVIEW, SIGNED_OFF, OVERRIDDEN
            "assigned_radiologist": study_data.get("assigned_radiologist", "Unassigned"),
            "patient_info": study_data.get("patient_info", {}),
            "clinical_indication": study_data.get("clinical_indication", ""),
            "modality": study_data.get("modality", "CR"),
            "body_part": study_data.get("body_part", "CHEST"),
            "dimensions": study_data.get("dimensions", []),
            "image_data_base64": study_data.get("image_data_base64"),
            "triage": study_data.get("triage", {
                "triage_priority": "ROUTINE",
                "risk_score": 0.12,
                "calibrated_confidence": 0.88,
                "triage_rationale": "Standard order"
            }),
            "ai_result": study_data.get("ai_result", {}),
            "review_record": study_data.get("review_record", None)
        }
        cls._studies[study_id] = record
        return record

    @classmethod
    def get_study(cls, study_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single study by ID."""
        return cls._studies.get(study_id)

    @classmethod
    def list_worklist(
        cls,
        priority_filter: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns worklist sorted by clinical urgency:
        1. STAT (Emergency, < 15 min SLA)
        2. URGENT (Urgent, < 60 min SLA)
        3. ROUTINE (Routine, < 240 min SLA)
        """
        priority_order = {"STAT": 0, "URGENT": 1, "ROUTINE": 2}
        results = list(cls._studies.values())

        if priority_filter and priority_filter.upper() != "ALL":
            results = [s for s in results if s["triage"]["triage_priority"].upper() == priority_filter.upper()]

        if status_filter and status_filter.upper() != "ALL":
            results = [s for s in results if s["status"].upper() == status_filter.upper()]

        # Sort primarily by priority score and secondarily by creation timestamp
        results.sort(key=lambda s: (
            priority_order.get(s["triage"]["triage_priority"], 3),
            -s["triage"].get("risk_score", 0.0)
        ))
        return results

    @classmethod
    def update_status(cls, study_id: str, status: str, radiologist: Optional[str] = None) -> bool:
        """Updates clinical workflow status of a study."""
        if study_id in cls._studies:
            cls._studies[study_id]["status"] = status
            if radiologist:
                cls._studies[study_id]["assigned_radiologist"] = radiologist
            return True
        return False
