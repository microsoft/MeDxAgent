"""Conversation/dialog data models."""

from pydantic import BaseModel, Field


class DialogTurn(BaseModel):
    """A single turn in the doctor-patient dialog."""
    
    role: str = Field(..., description="'D' for doctor, 'P' for patient")
    content: str = Field(..., description="The question or answer content")
    is_idk: bool = Field(default=False, description="True if this is an 'I don't know' answer")
    # For doctor turns - question reasoning (simple prompts only)
    question_reasoning: str | None = Field(default=None, description="Rationale for the doctor's question (simple prompts only)")
    disease: str | None = Field(default=None, description="Predicted disease after this turn")
    confidence: int | None = Field(default=None, description="Confidence score after this turn")
    # For differential-diagnosis workflows — store all 3 top predictions
    final_predictions: list[dict] | None = Field(default=None, description="Final merged top predictions with disease and confidence")
    # For KB-integrated workflows — store the per-source predictions before merging
    llm_predictions: str | None = Field(default=None, description="LLM diagnosis predictions as compact JSON string")
    kb_predictions: str | None = Field(default=None, description="KB diagnosis predictions as compact JSON string")
    # For specialist-ensemble workflows — all specialty predictions (disease + confidence only)
    specialist_predictions: str | None = Field(default=None, description="Per-specialty predictions as compact JSON string {specialty: [{disease, confidence}, ...]}")
    # For evidence-gap workflows — newline-joined gaps logged per turn
    general_evidence_gaps: str | None = Field(default=None, description="General evidence gaps, one per line")
    differential_evidence_gaps: str | None = Field(default=None, description="Differential evidence gaps, one 'Disease: missing_evidence' per line")
    # Prediction status - success or error message
    prediction_status: str | None = Field(default=None, description="'success' or error message if prediction failed")
    
    def format(self) -> str:
        """Format the turn as 'D': <content> or 'P': <content>."""
        return f"'{self.role}': {self.content}"


class Dialog(BaseModel):
    """The complete doctor-patient dialog history."""
    
    turns: list[DialogTurn] = Field(default_factory=list, description="List of dialog turns")
    
    def add_doctor_question(self, question: str, question_reasoning: str | None = None) -> None:
        """Add a doctor's question to the dialog."""
        self.turns.append(DialogTurn(role="D", content=question, question_reasoning=question_reasoning))
    
    def add_patient_answer(self, answer: str) -> None:
        """Add a patient's answer to the dialog."""
        self.turns.append(DialogTurn(role="P", content=answer))
    
    def add_patient_idk_answer(self, answer: str) -> None:
        """Add a patient's 'I don't know' answer to the dialog.
        
        This keeps the answer in history (so the same Q isn't asked again)
        but marks it so it won't count toward the round count.
        """
        self.turns.append(DialogTurn(role="P", content=answer, is_idk=True))
    
    def update_last_turn_diagnosis(self, disease: str, confidence: int) -> None:
        """Update the last patient turn with diagnosis information.
        
        Should be called after a diagnosis is made following a patient answer.
        """
        for turn in reversed(self.turns):
            if turn.role == "P" and not turn.is_idk:
                turn.disease = disease
                turn.confidence = confidence
                break
    
    def update_last_turn_differential_diagnosis(
        self,
        disease: str,
        confidence: int,
        final_predictions: list[dict],
        llm_predictions: str | None = None,
        kb_predictions: str | None = None
    ) -> None:
        """Update the last patient turn with all 3 predictions.

        Used by differential-diagnosis workflows to store all 3 top predictions
        (and, for KB-integrated workflows, the per-source LLM and KB predictions).

        Args:
            disease: Top predicted disease
            confidence: Top prediction confidence
            final_predictions: List of all 3 predictions with disease and confidence
            llm_predictions: Optional LLM predictions as compact JSON string
                (KB-integrated workflows only)
            kb_predictions: Optional KB predictions as compact JSON string
                (KB-integrated workflows only)
        """
        for turn in reversed(self.turns):
            if turn.role == "P" and not turn.is_idk:
                turn.disease = disease
                turn.confidence = confidence
                turn.final_predictions = final_predictions
                turn.llm_predictions = llm_predictions
                turn.kb_predictions = kb_predictions
                break
    
    def update_last_turn_prediction_status(self, status: str) -> None:
        """Update the last patient turn with prediction status.
        
        Args:
            status: 'success' or error message describing why prediction failed
        """
        for turn in reversed(self.turns):
            if turn.role == "P" and not turn.is_idk:
                turn.prediction_status = status
                break
    
    def update_last_turn_specialist_predictions(self, predictions_json: str) -> None:
        """Update the last patient turn with specialist-ensemble predictions.
        
        Args:
            predictions_json: Compact JSON string mapping specialty -> list of
                {disease, confidence} dicts.
        """
        for turn in reversed(self.turns):
            if turn.role == "P" and not turn.is_idk:
                turn.specialist_predictions = predictions_json
                break
    
    def update_last_turn_evidence_gaps(
        self,
        general: str | None = None,
        differential: str | None = None,
    ) -> None:
        """Update the last patient turn with evidence-gap report fields.
        
        Args:
            general: Newline-joined string of general gaps (one per line).
            differential: Newline-joined string of 'Disease: missing_evidence'
                (one per line).
        """
        for turn in reversed(self.turns):
            if turn.role == "P" and not turn.is_idk:
                if general is not None:
                    turn.general_evidence_gaps = general
                if differential is not None:
                    turn.differential_evidence_gaps = differential
                break
    
    def format(self) -> str:
        """Format the entire dialog as a string.
        
        Returns:
            String in format: 'D': <q>, 'P': <ans>, 'D': <q>...
        """
        return ", ".join(turn.format() for turn in self.turns)
    
    def format_readable(self) -> str:
        """Format the dialog in a more readable format with newlines."""
        lines = []
        for turn in self.turns:
            prefix = "Doctor" if turn.role == "D" else "Patient"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)
    
    def format_readable_no_idk(self) -> str:
        """Format the dialog excluding IDK Q&A pairs."""
        lines = []
        skip_next_patient = False
        i = 0
        while i < len(self.turns):
            turn = self.turns[i]
            if turn.role == "D":
                # Check if the next turn is an IDK patient answer
                next_is_idk = (
                    i + 1 < len(self.turns)
                    and self.turns[i + 1].role == "P"
                    and self.turns[i + 1].is_idk
                )
                if not next_is_idk:
                    lines.append(f"Doctor: {turn.content}")
                else:
                    i += 2  # Skip both the doctor Q and the IDK answer
                    continue
            elif turn.role == "P" and not turn.is_idk:
                lines.append(f"Patient: {turn.content}")
            i += 1
        return "\n".join(lines)
    
    def get_doctor_questions(self) -> list[str]:
        """Get a list of all doctor questions asked so far.
        
        Returns:
            List of question strings (just the content, no prefixes)
        """
        return [turn.content for turn in self.turns if turn.role == "D"]

    def format_structured(self) -> list[dict]:
        """Format the dialog as a list of structured round objects.
        
        Returns:
            List of dicts with {"round": n, "doctor": question, "patient": answer}
        """
        structured = []
        current_round = 0
        current_entry = {}
        
        for turn in self.turns:
            if turn.role == "D":
                # Start a new round
                if current_entry:  # Save previous incomplete entry
                    structured.append(current_entry)
                current_round += 1
                current_entry = {"round": current_round, "doctor": turn.content}
                # Include question reasoning if present (simple prompts)
                if turn.question_reasoning is not None:
                    current_entry["question_reasoning"] = turn.question_reasoning
            elif turn.role == "P":
                if current_entry:
                    current_entry["patient"] = turn.content
                    if turn.disease is not None:
                        current_entry["disease"] = turn.disease
                    if turn.confidence is not None:
                        current_entry["confidence"] = turn.confidence
                    # KB-integrated workflows — KB-only candidate predictions
                    if turn.kb_predictions is not None:
                        current_entry["kb_predictions"] = turn.kb_predictions
                    # Specialist-ensemble workflows — per-specialty predictions
                    if turn.specialist_predictions is not None:
                        current_entry["specialist_predictions"] = turn.specialist_predictions
                    # KB-integrated workflows — LLM-only predictions before merging
                    if turn.llm_predictions is not None:
                        current_entry["llm_predictions"] = turn.llm_predictions
                    # Final merged top predictions for this turn
                    if turn.final_predictions is not None:
                        current_entry["final_predictions"] = turn.final_predictions
                    # Evidence-gap workflows — include per-turn gap reports
                    if turn.general_evidence_gaps is not None:
                        current_entry["general_evidence_gaps"] = turn.general_evidence_gaps
                    if turn.differential_evidence_gaps is not None:
                        current_entry["differential_evidence_gaps"] = turn.differential_evidence_gaps
                    # Include prediction status if set
                    if turn.prediction_status is not None:
                        current_entry["prediction_status"] = turn.prediction_status
                    structured.append(current_entry)
                    current_entry = {}
        
        # Handle any remaining entry
        if current_entry:
            structured.append(current_entry)
        
        return structured
    
    def get_last_question(self) -> str | None:
        """Get the last doctor's question."""
        for turn in reversed(self.turns):
            if turn.role == "D":
                return turn.content
        return None
    
    def round_count(self) -> int:
        """Count the number of complete dialog rounds (Q&A pairs), excluding IDK answers."""
        doctor_count = sum(1 for turn in self.turns if turn.role == "D")
        # Only count patient answers that are not "I don't know"
        patient_count = sum(1 for turn in self.turns if turn.role == "P" and not turn.is_idk)
        return min(doctor_count, patient_count)
    
    def copy(self) -> "Dialog":
        """Create a deep copy of the dialog."""
        return Dialog(turns=[
            DialogTurn(
                role=t.role, content=t.content, is_idk=t.is_idk,
                disease=t.disease, confidence=t.confidence,
                final_predictions=t.final_predictions,
            ) for t in self.turns
        ])
