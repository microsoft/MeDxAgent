"""Knowledge graph integrations for medical diagnosis."""

from src.knowledge_graphs.umls_rest_client import UMLSClient
from src.knowledge_graphs.kg_symptom_lookup import (
    extract_symptoms_from_summary,
    lookup_diseases_from_symptoms,
    format_kb_candidates_for_agent,
    get_kb_disease_candidates,
)

__all__ = [
    "UMLSClient",
    "extract_symptoms_from_summary",
    "lookup_diseases_from_symptoms",
    "format_kb_candidates_for_agent",
    "get_kb_disease_candidates",
]
