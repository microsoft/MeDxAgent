"""Agent modules for the medical diagnosis simulation system."""

from .base_agent import BaseAgent
from .patient_agent import PatientAgent
from .doctor_q_agent import DoctorQAgent
from .doctor_q_diff_agent import DoctorQDiffAgent
from .summarizer_agent import SummarizerAgent
from .summarizer_str_agent import SummarizerStrAgent
from .diagnosis_dialog_agent import DiagnosisDialogAgent
from .diagnosis_summary_agent import DiagnosisSummaryAgent
from .diagnosis_kb_agent import DiagnosisKBAgent
from .diagnosis_merge_agent import DiagnosisMergeAgent
from .judge_agent import JudgeAgent
from .evidence_gap_agent import EvidenceGapAgent
from .doctor_q_diff_evidence_agent import DoctorQDiffEvidenceAgent
from .doctor_q_evidence_agent import DoctorQEvidenceAgent
from .specialist_ensemble_agent import SpecialistEnsembleAgent

AGENT_REGISTRY = {
    "patient_agent": PatientAgent,
    "doctor_q_agent": DoctorQAgent,
    "doctor_q_diff_agent": DoctorQDiffAgent,
    "summarizer_agent": SummarizerAgent,
    "summarizer_str_agent": SummarizerStrAgent,
    "diagnosis_dialog_agent": DiagnosisDialogAgent,
    "diagnosis_summary_agent": DiagnosisSummaryAgent,
    "diagnosis_kb_agent": DiagnosisKBAgent,
    "diagnosis_merge_agent": DiagnosisMergeAgent,
    "judge_agent": JudgeAgent,
    "evidence_gap_agent": EvidenceGapAgent,
    "doctor_q_diff_evidence_agent": DoctorQDiffEvidenceAgent,
    "doctor_q_evidence_agent": DoctorQEvidenceAgent,
    "specialist_ensemble_agent": SpecialistEnsembleAgent,
}


def get_agent(name: str) -> type[BaseAgent]:
    """Get agent class by name."""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[name]


__all__ = [
    "BaseAgent",
    "PatientAgent",
    "DoctorQAgent",
    "DoctorQDiffAgent",
    "SummarizerAgent",
    "SummarizerStrAgent",
    "DiagnosisDialogAgent",
    "DiagnosisSummaryAgent",
    "DiagnosisKBAgent",
    "DiagnosisMergeAgent",
    "JudgeAgent",
    "EvidenceGapAgent",
    "DoctorQDiffEvidenceAgent",
    "AGENT_REGISTRY",
    "get_agent",
]
