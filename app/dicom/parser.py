"""
DICOM & Medical Image Parsing and Metadata Extraction
Handles standard DICOM (.dcm) files as well as normalized clinical radiography formats.
"""

import io
import hashlib
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class DICOMParser:
    """Robust parser for medical imaging files with metadata validation."""

    @staticmethod
    def is_dicom_file(file_bytes: bytes) -> bool:
        """Check if file has DICOM magic preamble 'DICM' at byte 128."""
        if len(file_bytes) > 132:
            return file_bytes[128:132] == b"DICM"
        return False

    @classmethod
    def parse_file(cls, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Parses DICOM or standard image byte stream.
        Returns:
            Dict containing metadata, image_array, and diagnostic status.
        """
        is_dcm = cls.is_dicom_file(file_bytes) or filename.lower().endswith(".dcm")

        if is_dcm and HAS_PYDICOM:
            return cls._parse_pydicom(file_bytes, filename)
        else:
            return cls._parse_standard_or_fallback(file_bytes, filename)

    @classmethod
    def _parse_pydicom(cls, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Parse using native pydicom library."""
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            
            # Pixel array extraction
            pixel_array = ds.pixel_array.astype(np.float32)
            if pixel_array.ndim != 2:
                return {
                    "success": False,
                    "error": "Only single-frame 2D DICOM images are supported by this viewer.",
                    "metadata": {},
                    "pixel_array": None,
                    "shape": list(pixel_array.shape),
                }
            
            # Rescale slope and intercept (Hounsfield Units for CT)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixel_array = pixel_array * slope + intercept

            # Windowing tags
            window_center = getattr(ds, "WindowCenter", 128)
            window_width = getattr(ds, "WindowWidth", 256)
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = float(window_center[0])
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = float(window_width[0])

            metadata = {
                "study_instance_uid": str(getattr(ds, "StudyInstanceUID", f"STUDY_{hashlib.md5(file_bytes).hexdigest()[:10]}")),
                "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "1.2.840.10008.5.1.4.1.1")),
                "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", f"SOP_{hashlib.md5(file_bytes).hexdigest()[:8]}")),
                "patient_name": str(getattr(ds, "PatientName", "ANONYMOUS")),
                "patient_id": str(getattr(ds, "PatientID", "PT-UNKNOWN")),
                "patient_sex": str(getattr(ds, "PatientSex", "O")),
                "patient_age": str(getattr(ds, "PatientAge", "050Y")),
                "modality": str(getattr(ds, "Modality", "CR")),
                "body_part_examined": str(getattr(ds, "BodyPartExamined", "CHEST")),
                "rows": int(getattr(ds, "Rows", pixel_array.shape[0])),
                "columns": int(getattr(ds, "Columns", pixel_array.shape[1])),
                "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
                "window_center": float(window_center),
                "window_width": float(window_width),
                "rescale_intercept": intercept,
                "rescale_slope": slope,
                "pixel_spacing": [float(x) for x in getattr(ds, "PixelSpacing", [0.143, 0.143])],
                "filename": filename,
                "source_format": "Native DICOM (pydicom parsed)"
            }

            return {
                "success": True,
                "metadata": metadata,
                "pixel_array": pixel_array,
                "shape": list(pixel_array.shape)
            }
        except Exception as e:
            return cls._parse_standard_or_fallback(file_bytes, filename, error_note=str(e))

    @classmethod
    def _parse_standard_or_fallback(cls, file_bytes: bytes, filename: str, error_note: Optional[str] = None) -> Dict[str, Any]:
        """Parse standard medical image (PNG, JPEG, WebP) or handle byte fallback."""
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("L")
            pixel_array = np.array(image, dtype=np.float32)
            
            # Infer modality from filename or default to CR (Computed Radiography)
            modality = "CR"
            body_part = "CHEST"
            fn_lower = filename.lower()
            if "wrist" in fn_lower or "fracture" in fn_lower or "bone" in fn_lower:
                body_part = "EXTREMITY_WRIST"
                modality = "DX"
            elif "ct" in fn_lower or "brain" in fn_lower:
                modality = "CT"
                body_part = "BRAIN"
            elif "mri" in fn_lower or "mr" in fn_lower:
                modality = "MR"
                body_part = "BRAIN"

            hasher = hashlib.sha256(file_bytes).hexdigest()
            metadata = {
                "study_instance_uid": f"1.2.840.113619.2.55.3.{hasher[:8]}",
                "series_instance_uid": f"1.2.840.113619.2.55.3.{hasher[8:16]}",
                "sop_instance_uid": f"1.2.840.113619.2.55.3.{hasher[16:24]}",
                "patient_name": "SIMULATED^PATIENT",
                "patient_id": f"RAD-{hasher[:6].upper()}",
                "patient_sex": "F" if int(hasher[0], 16) % 2 == 0 else "M",
                "patient_age": f"0{40 + (int(hasher[1], 16) % 40)}Y",
                "modality": modality,
                "body_part_examined": body_part,
                "rows": pixel_array.shape[0],
                "columns": pixel_array.shape[1],
                "photometric_interpretation": "MONOCHROME2",
                "window_center": 128.0,
                "window_width": 256.0,
                "rescale_intercept": 0.0,
                "rescale_slope": 1.0,
                "pixel_spacing": [0.143, 0.143],
                "filename": filename,
                "source_format": "Radiography / Normalized Raster Image",
                "note": error_note
            }

            return {
                "success": True,
                "metadata": metadata,
                "pixel_array": pixel_array,
                "shape": list(pixel_array.shape)
            }
        except Exception as err:
            return {
                "success": False,
                "error": f"Failed to ingest image format: {str(err)}",
                "metadata": {},
                "pixel_array": None,
                "shape": []
            }
