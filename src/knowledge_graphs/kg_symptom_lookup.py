"""Utility functions for knowledge graph-based symptom-to-disease lookup."""

import json
from collections import Counter
from typing import Optional

from src.knowledge_graphs.umls_rest_client import UMLSClient


def extract_symptoms_from_summary(str_summary: dict | str) -> list[str]:
    """Extract symptom terms from a structured clinical summary.
    
    Args:
        str_summary: The structured summary (dict or JSON string)
        
    Returns:
        List of symptom terms to query against the knowledge graph
    """
    if isinstance(str_summary, str):
        try:
            str_summary = json.loads(str_summary)
        except json.JSONDecodeError:
            # If it's not valid JSON, try to extract keywords
            return []
    
    symptoms = []
    
    # Extract from symptoms_present field
    if "symptoms_present" in str_summary:
        symptoms.extend(str_summary["symptoms_present"])
    
    # Extract from physical_exam findings (can indicate symptoms)
    if "physical_exam" in str_summary:
        symptoms.extend(str_summary["physical_exam"])
    
    # Extract from vital_signs if abnormal (the values themselves are informative)
    if "vital_signs" in str_summary:
        for vital in str_summary["vital_signs"]:
            # Keep vital sign descriptions as they may indicate conditions
            symptoms.append(vital)
    
    # Filter out empty strings and duplicates while preserving order
    seen = set()
    unique_symptoms = []
    for symptom in symptoms:
        if symptom and symptom not in seen:
            seen.add(symptom)
            unique_symptoms.append(symptom)
    
    return unique_symptoms


def lookup_diseases_from_symptoms(
    symptoms: list[str],
    umls_client: Optional[UMLSClient] = None
) -> list[dict]:
    """Look up diseases associated with a list of symptoms using UMLS.
    
    Returns a union of all diseases found, with a count of how many symptoms
    each disease matches.
    
    Args:
        symptoms: List of symptom terms to look up
        umls_client: Optional pre-initialized UMLS client
        
    Returns:
        List of disease candidates with symptom match counts, sorted by match count.
        Each dict has: disease_name, disease_cui, symptom_match_count, matched_symptoms
    """
    if not symptoms:
        return []
    
    # Initialize client if not provided
    if umls_client is None:
        try:
            umls_client = UMLSClient()
        except ValueError as e:
            # No API key available
            print(f"UMLS client initialization failed: {e}")
            return []
    
    # Track diseases and which symptoms they match
    disease_symptom_matches: dict[str, dict] = {}  # disease_cui -> info
    
    api_errors = 0
    
    for symptom in symptoms:
        try:
            diseases = umls_client.get_diseases_from_symptom(symptom)
            
            if diseases:
                print(f"    [KB] '{symptom}' → {len(diseases)} diseases (top: {', '.join(d['disease_name'] for d in diseases[:3])})")
            else:
                print(f"    [KB] '{symptom}' → 0 diseases")
            
            for disease in diseases:
                disease_cui = disease.get("disease_cui")
                disease_name = disease.get("disease_name", "Unknown")
                
                if not disease_cui:
                    continue
                
                if disease_cui not in disease_symptom_matches:
                    disease_symptom_matches[disease_cui] = {
                        "disease_cui": disease_cui,
                        "disease_name": disease_name,
                        "matched_symptoms": set(),
                        "sources": set()
                    }
                
                disease_symptom_matches[disease_cui]["matched_symptoms"].add(symptom)
                if disease.get("source"):
                    disease_symptom_matches[disease_cui]["sources"].add(disease.get("source"))
                    
        except Exception as e:
            # Retry once before counting as error
            import asyncio
            print(f"  [KB RETRY] symptom '{symptom}' failed: {str(e)[:80]}... retrying")
            try:
                import time
                time.sleep(2)
                diseases = umls_client.get_diseases_from_symptom(symptom)
                if diseases:
                    print(f"    [KB] '{symptom}' → {len(diseases)} diseases (retry success)")
                else:
                    print(f"    [KB] '{symptom}' → 0 diseases (retry success)")
                
                for disease in diseases:
                    disease_cui = disease.get("disease_cui")
                    disease_name = disease.get("disease_name", "Unknown")
                    if not disease_cui:
                        continue
                    if disease_cui not in disease_symptom_matches:
                        disease_symptom_matches[disease_cui] = {
                            "disease_cui": disease_cui,
                            "disease_name": disease_name,
                            "matched_symptoms": set(),
                            "sources": set()
                        }
                    disease_symptom_matches[disease_cui]["matched_symptoms"].add(symptom)
                    if disease.get("source"):
                        disease_symptom_matches[disease_cui]["sources"].add(disease.get("source"))
            except Exception as e2:
                api_errors += 1
                print(f"  ████ KB API ERROR ████ symptom '{symptom}' (retry also failed): {e2}")
                continue
    
    # If any symptom lookups failed, raise an error so the workflow logs it
    if api_errors > 0:
        raise RuntimeError(f"KB API failed for {api_errors}/{len(symptoms)} symptoms (403/connection errors)")
    
    # Convert to list format with match counts
    result = []
    for cui, info in disease_symptom_matches.items():
        result.append({
            "disease_name": info["disease_name"],
            "disease_cui": cui,
            "symptom_match_count": len(info["matched_symptoms"]),
            "matched_symptoms": list(info["matched_symptoms"]),
            "sources": list(info["sources"])
        })
    
    # Sort by match count (descending), then by name for consistency
    result.sort(key=lambda x: (-x["symptom_match_count"], x["disease_name"]))
    
    return result


def format_kb_candidates_for_agent(kb_candidates: list[dict], max_candidates: int = 10000) -> str:
    """Format KB disease candidates for the diagnosis_kb_agent prompt.
    
    Args:
        kb_candidates: List of disease candidates from lookup_diseases_from_symptoms
        max_candidates: Maximum number of candidates to include
        
    Returns:
        Formatted string for inclusion in agent prompt
    """
    if not kb_candidates:
        return "No disease candidates found in knowledge base."
    
    # Limit to top N candidates
    candidates = kb_candidates[:max_candidates]
    
    lines = []
    for i, candidate in enumerate(candidates, 1):
        name = candidate["disease_name"].title()  # Normalize to title case
        count = candidate["symptom_match_count"]
        symptoms = ", ".join(candidate["matched_symptoms"])
        lines.append(f"{i}. {name} [matches {count} symptom(s): {symptoms}]")
    
    return "\n".join(lines)


def get_kb_disease_candidates(
    str_summary: dict | str,
    umls_client: Optional[UMLSClient] = None,
    max_candidates: int = 10000
) -> tuple[list[dict], str]:
    """Main entry point: get KB disease candidates from a structured summary.
    
    Args:
        str_summary: The structured clinical summary
        umls_client: Optional pre-initialized UMLS client
        max_candidates: Maximum candidates to return
        
    Returns:
        Tuple of (raw candidates list, formatted string for agent)
    """
    print("Extracting symptoms from summary for KB lookup...")
    # print("Summary input:", str_summary)
    symptoms = extract_symptoms_from_summary(str_summary)
    print("\nExtracted symptoms:", symptoms)
    # print("\nLooking up diseases from symptoms in KB...")
    
    if not symptoms:
        return [], "No symptoms extracted from summary for KB lookup."
    
    candidates = lookup_diseases_from_symptoms(symptoms, umls_client)
    formatted = format_kb_candidates_for_agent(candidates, max_candidates)
    # print("TESTING\n")
    # print(formatted)
    # print("\nEND")
    
    return candidates, formatted
