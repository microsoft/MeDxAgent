"""Data models for the medical diagnosis simulation system."""

from .patient_case import PatientCase, PatientCaseFile
from .conversation import Dialog, DialogTurn
from .diagnosis import DiagnosisResult

__all__ = [
    "PatientCase",
    "PatientCaseFile",
    "Dialog",
    "DialogTurn",
    "DiagnosisResult",
]
