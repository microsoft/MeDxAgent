"""Workflow context for sharing state between agents."""

from typing import Any
from src.models.conversation import Dialog


class WorkflowContext:
    """Shared context for workflow execution.
    
    Stores and manages state that is passed between agents during
    workflow execution, including dialog history, summaries, and
    intermediate results.
    """
    
    def __init__(self):
        self._data: dict[str, Any] = {}
        self._dialog: Dialog = Dialog()
        self._trace: list[dict[str, Any]] = []
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self._data.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the context."""
        return key in self._data
    
    def delete(self, key: str) -> None:
        """Delete a key from the context."""
        if key in self._data:
            del self._data[key]
    
    @property
    def dialog(self) -> Dialog:
        """Get the dialog object."""
        return self._dialog
    
    def add_doctor_question(self, question: str, question_reasoning: str | None = None) -> None:
        """Add a doctor's question to the dialog."""
        self._dialog.add_doctor_question(question, question_reasoning=question_reasoning)
    
    def add_patient_answer(self, answer: str) -> None:
        """Add a patient's answer to the dialog."""
        self._dialog.add_patient_answer(answer)
    
    def add_patient_idk_answer(self, answer: str) -> None:
        """Add a patient's 'I don't know' answer to the dialog.
        
        Keeps the answer in history (so the same Q isn't asked again)
        but won't count toward the round count.
        """
        self._dialog.add_patient_idk_answer(answer)
    
    def update_last_turn_diagnosis(self, disease: str, confidence: int) -> None:
        """Update the last patient turn with diagnosis information."""
        self._dialog.update_last_turn_diagnosis(disease, confidence)
    
    def update_last_turn_differential_diagnosis(
        self,
        disease: str,
        confidence: int,
        final_predictions: list[dict],
        llm_predictions: str | None = None,
        kb_predictions: str | None = None
    ) -> None:
        """Update the last patient turn with all 3 predictions."""
        self._dialog.update_last_turn_differential_diagnosis(
            disease, confidence, final_predictions, llm_predictions, kb_predictions
        )
    
    def update_last_turn_prediction_status(self, status: str) -> None:
        """Update the last patient turn with prediction status."""
        self._dialog.update_last_turn_prediction_status(status)
    
    def update_last_turn_specialist_predictions(self, predictions_json: str) -> None:
        """Update the last patient turn with specialist-ensemble predictions (compact JSON)."""
        self._dialog.update_last_turn_specialist_predictions(predictions_json)
    
    def update_last_turn_evidence_gaps(
        self,
        general: str | None = None,
        differential: str | None = None,
    ) -> None:
        """Update the last patient turn with evidence-gap report fields."""
        self._dialog.update_last_turn_evidence_gaps(general=general, differential=differential)
    
    def get_dialog_string(self) -> str:
        """Get the formatted dialog string."""
        return self._dialog.format()
    
    def get_dialog_readable(self) -> str:
        """Get the dialog in readable format."""
        return self._dialog.format_readable()
    
    def get_dialog_readable_no_idk(self) -> str:
        """Get the dialog in readable format, excluding IDK Q&A pairs."""
        return self._dialog.format_readable_no_idk()
    
    def get_doctor_questions(self) -> list[str]:
        """Get list of all doctor questions asked so far."""
        return self._dialog.get_doctor_questions()
    
    def get_dialog_structured(self) -> list[dict]:
        """Get the dialog as structured list of round objects."""
        return self._dialog.format_structured()
    
    def get_round_count(self) -> int:
        """Get the number of complete dialog rounds."""
        return self._dialog.round_count()
    
    def add_trace(self, agent_name: str, action: str, data: dict[str, Any]) -> None:
        """Add an entry to the execution trace."""
        self._trace.append({
            "agent": agent_name,
            "action": action,
            "round": self.get_round_count(),
            **data
        })
    
    def get_trace(self) -> list[dict[str, Any]]:
        """Get the execution trace."""
        return self._trace
    
    def get_confidence_progression(self) -> list[int]:
        """Extract confidence scores from the trace."""
        confidences = []
        for entry in self._trace:
            if "confidence" in entry:
                confidences.append(entry["confidence"])
        return confidences
    
    def get_diagnosis_progression(self) -> list[dict]:
        """Extract disease predictions and confidence scores from the trace.
        
        Returns a list of dicts with 'disease', 'confidence', and 'round' for each diagnosis attempt.
        Useful for debugging to see if a correct diagnosis was made in an earlier turn.
        """
        progression = []
        for entry in self._trace:
            if "disease" in entry and "confidence" in entry:
                progression.append({
                    "round": entry.get("round", 0),
                    "disease": entry["disease"],
                    "confidence": entry["confidence"]
                })
        return progression
    
    def reset(self) -> None:
        """Reset the context for a new case."""
        self._data = {}
        self._dialog = Dialog()
        self._trace = []
    
    def copy(self) -> "WorkflowContext":
        """Create a copy of the context."""
        new_context = WorkflowContext()
        new_context._data = self._data.copy()
        new_context._dialog = self._dialog.copy()
        new_context._trace = self._trace.copy()
        return new_context
