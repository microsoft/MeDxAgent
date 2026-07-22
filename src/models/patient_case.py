"""Patient case data models."""

from pydantic import BaseModel, Field, model_validator
from typing import Any


class PatientCase(BaseModel):
    """A single patient case with history and ground truth diagnosis."""
    
    case_id: str = Field(..., description="Unique identifier for the case")
    patient_history: str = Field(default="", description="Detailed patient history including symptoms, family history, physical examination, and lab/imaging results")
    gt: str | list[str] = Field(..., description="Ground truth disease diagnosis (single or multiple)")
    
    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        """Normalize field names from different dataset formats."""
        if isinstance(data, dict):
            # Alias: case_vignette -> patient_history
            if 'case_vignette' in data and 'patient_history' not in data:
                data['patient_history'] = data.pop('case_vignette')
            
            # Alias: gt_diagnosis -> gt
            if 'gt_diagnosis' in data and 'gt' not in data:
                data['gt'] = data.pop('gt_diagnosis')
        return data
    
    # Optional metadata
    age: int | None = Field(None, description="Patient age")
    gender: str | None = Field(None, description="Patient gender")
    chief_complaint: str | None = Field(None, description="Primary reason for visit")


class PatientCaseFile(BaseModel):
    """Container for multiple patient cases loaded from JSON."""
    
    cases: list[PatientCase] = Field(..., description="List of patient cases")
    
    @classmethod
    def from_json_file(cls, file_path: str) -> "PatientCaseFile":
        """Load patient cases from a JSON file."""
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list format and dict format with 'cases' key
        if isinstance(data, list):
            return cls(cases=[PatientCase(**case) for case in data])
        elif isinstance(data, dict) and "cases" in data:
            return cls(cases=[PatientCase(**case) for case in data["cases"]])
        else:
            raise ValueError("Invalid JSON format. Expected list or dict with 'cases' key.")
