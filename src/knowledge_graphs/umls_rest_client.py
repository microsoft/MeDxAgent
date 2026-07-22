"""UMLS REST API Client for symptom-disease relationship lookup.

Uses the official NLM UMLS REST API (https://uts-ws.nlm.nih.gov/rest)
as a replacement for the UBKG API which was unreliable.

API Documentation: https://documentation.uts.nlm.nih.gov/rest/home.html
"""

import os
import requests
from typing import Optional


BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
UMLS_VERSION = "current"

# Semantic types that represent diseases/disorders
DISEASE_SEMANTIC_TYPES = {
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T019": "Congenital Abnormality",
    "T191": "Neoplastic Process",
    "T046": "Pathologic Function",
    "T190": "Anatomical Abnormality",
    "T049": "Cell or Molecular Dysfunction",
    "T037": "Injury or Poisoning",
    "T033": "Finding",
}

# Relationship labels that indicate symptom→disease associations
DISEASE_REL_LABELS = {
    "manifestation_of", "may_be_finding_of_disease",
    "associated_with", "inverse_isa",
    "cause_of", "has_causative_agent",
}

# Only keep results from English / major biomedical sources
RELEVANT_SOURCES = {
    "OMIM", "NCI", "SNOMEDCT_US", "MSH", "MEDCIN", "HPO",
    "ICD10CM", "ICD10", "ICD9CM", "NDFRT", "RXNORM",
    "CSP", "CCSR_ICD10CM", "MDR",
}


class UMLSClient:
    """UMLS REST API client for symptom→disease lookup.

    Uses the official NLM UMLS REST API (https://uts-ws.nlm.nih.gov/rest).
    """

    def __init__(self, umls_api_key: Optional[str] = None):
        """Initialize the UMLS REST client.
        
        Args:
            umls_api_key: UMLS API key. If not provided, reads UMLS_API_KEY env var.
        """
        self.api_key = umls_api_key or os.getenv("UMLS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "UMLS API key is required. Provide it as a parameter or set "
                "the UMLS_API_KEY environment variable."
            )
        self.session = requests.Session()

    def _get(self, url: str, params: dict | None = None, timeout: int = 30) -> dict:
        """Make a GET request to the UMLS REST API.
        
        Args:
            url: Full URL to request
            params: Query parameters (apiKey is added automatically)
            timeout: Request timeout in seconds
            
        Returns:
            Parsed JSON response
            
        Raises:
            requests.exceptions.HTTPError: On non-2xx responses
        """
        params = params or {}
        params["apiKey"] = self.api_key
        resp = self.session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _search_term(self, term: str, search_type: str = "exact") -> list[dict]:
        """Search UMLS for a term and return matching concepts.
        
        Args:
            term: The term to search for (e.g., "cough", "fever")
            search_type: "exact" or "words"
            
        Returns:
            List of result dicts with keys: ui (CUI), name, rootSource, uri
        """
        url = f"{BASE_URL}/search/{UMLS_VERSION}"
        params = {
            "string": term,
            "returnIdType": "concept",
            "searchType": search_type,
            "pageSize": 10,
        }
        
        data = self._get(url, params)
        results = data.get("result", {}).get("results", [])
        
        # Filter out NONE placeholder results
        results = [r for r in results if r.get("ui") != "NONE"]
        
        # If exact match fails, try word search
        if not results and search_type == "exact":
            return self._search_term(term, search_type="words")
        
        return results

    def _get_relations(self, cui: str, max_pages: int = 5) -> list[dict]:
        """Get all relations for a CUI with pagination.
        
        Args:
            cui: Concept Unique Identifier (e.g., "C0010200")
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of relation dicts with keys: relationLabel, additionalRelationLabel,
            relatedIdName, relatedId, rootSource, etc.
        """
        all_relations = []
        
        for page in range(1, max_pages + 1):
            url = f"{BASE_URL}/content/{UMLS_VERSION}/CUI/{cui}/relations"
            params = {"pageSize": 100, "pageNumber": page}
            
            try:
                data = self._get(url, params)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    break
                raise
            
            results = data.get("result", [])
            if not results:
                break
            
            all_relations.extend(results)
            
            # Stop if we got fewer than a full page (no more pages)
            if len(results) < 100:
                break
        
        return all_relations

    def _get_semantic_types(self, cui: str) -> list[str]:
        """Get semantic type TUIs for a concept.
        
        Args:
            cui: Concept Unique Identifier
            
        Returns:
            List of semantic type TUI strings (e.g., ["T047", "T191"])
        """
        url = f"{BASE_URL}/content/{UMLS_VERSION}/CUI/{cui}"
        
        try:
            data = self._get(url, timeout=15)
        except requests.exceptions.HTTPError:
            return []
        
        sem_types = data.get("result", {}).get("semanticTypes", [])
        # Each sem_type has a "uri" like ".../T047", extract the TUI
        return [st.get("uri", "").split("/")[-1] for st in sem_types]

    def get_diseases_from_symptom(self, symptom_name: str) -> list[dict]:
        """Get diseases associated with a given symptom.
        
        This is the main method used by the workflow. It:
        1. Searches UMLS for the symptom term → gets CUI
        2. Fetches all relations for that CUI
        3. Filters to disease concepts using relationship labels + semantic types
        
        Args:
            symptom_name: The symptom name (e.g., "cough", "fever", "chest pain")
            
        Returns:
            List of dicts with keys: disease_cui, disease_name, relationship, source
        """
        # Step 1: Search for symptom → CUI
        concepts = self._search_term(symptom_name)
        
        if not concepts:
            print(f"    ████ KB ERROR ████ No UMLS concepts found for symptom: '{symptom_name}'")
            return []
        
        # Use the top match
        top = concepts[0]
        cui = top["ui"]
        
        # Step 2: Get relations for this CUI
        try:
            relations = self._get_relations(cui)
        except Exception as e:
            print(f"    ████ KB ERROR ████ Failed to get relations for {cui} ({symptom_name}): {e}")
            return []
        
        if not relations:
            return []
        
        # Step 3: Filter to disease concepts
        diseases = []
        seen_names = set()
        
        for rel in relations:
            rel_label = rel.get("additionalRelationLabel", "")
            name = rel.get("relatedIdName", "").strip()
            source = rel.get("rootSource", "")
            related_id_url = rel.get("relatedId", "")
            # Extract CUI or code from the URL (last path segment)
            related_id = related_id_url.rstrip("/").split("/")[-1] if related_id_url else ""
            
            if not name or source not in RELEVANT_SOURCES:
                continue
            
            # Deduplicate by name
            name_key = name.upper()
            if name_key in seen_names:
                continue
            
            is_disease = False
            
            # Method 1: relationship label indicates disease association
            if rel_label in DISEASE_REL_LABELS:
                is_disease = True
            
            # Method 2: check semantic types for CUI-based IDs
            if not is_disease and related_id.startswith("C") and related_id[1:].isdigit():
                tuis = self._get_semantic_types(related_id)
                if any(t in DISEASE_SEMANTIC_TYPES for t in tuis):
                    is_disease = True
            
            if is_disease:
                seen_names.add(name_key)
                diseases.append({
                    "disease_cui": related_id,
                    "disease_name": name,
                    "relationship": rel_label or rel.get("relationLabel", ""),
                    "source": source,
                })
        
        return diseases
