"""Doctor Diagnosis Dialog Agent - Makes diagnosis from doctor-patient dialog."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.models.diagnosis import DiagnosisResult
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class DiagnosisPrediction(BaseModel):
    """A single diagnosis prediction with confidence."""
    disease: str = Field(..., description="Predicted disease name")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    reasoning: str = Field(default="", description="Brief reasoning for the diagnosis")


class DiagnosisPredictionReasoningFirst(BaseModel):
    """A single diagnosis prediction with reasoning BEFORE the disease name.
    
    Field order matters for structured output: the LLM generates fields in order,
    so placing reasoning first forces it to think before committing to a disease.
    """
    reasoning: str = Field(..., description="Key clinical findings and reasoning supporting this diagnosis")
    disease: str = Field(..., description="Predicted disease name")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")


class DiagnosisDialogResponse(BaseModel):
    """Structured response for diagnosis predictions from dialog."""
    predictions: list[DiagnosisPrediction] = Field(..., description="Top 3 disease predictions sorted by confidence")


class DiagnosisDialogResponseReasoningFirst(BaseModel):
    """Structured response with reasoning-first predictions."""
    predictions: list[DiagnosisPredictionReasoningFirst] = Field(..., description="Top 3 disease predictions with reasoning first")


class DiagnosisDialogAgentOutput(AgentOutput):
    """Output from the diagnosis dialog agent."""
    predictions: list[DiagnosisPrediction] = Field(default_factory=list, description="Top 3 disease predictions sorted by confidence")
    
    # Convenience properties for backward compatibility
    @property
    def disease(self) -> str:
        """Get the top predicted disease."""
        return self.predictions[0].disease if self.predictions else ""
    
    @property
    def confidence(self) -> int:
        """Get the top prediction's confidence."""
        return self.predictions[0].confidence if self.predictions else 0
    
    @property
    def reasoning(self) -> str:
        """Get the top prediction's reasoning."""
        return self.predictions[0].reasoning if self.predictions else ""


class DiagnosisDialogAgent(BaseAgent):
    """Agent that diagnoses disease based on doctor-patient dialog.

    The diagnosis agent:
    - Receives the complete doctor-patient dialog
    - Analyzes the information to make a diagnosis
    - Outputs the disease name and confidence level (0-100)
    - Confidence >= the workflow's ``high_confidence_threshold`` triggers the
      early-exit condition in the workflow loop
    - Can receive previous diagnosis to avoid repetition
    """
    
    agent_name = "diagnosis.dialog"
    
    SPECIALTY_AGENT_MAP = {
        "Cardiology": "diagnosis.dialog_cardiology",
        "Pulmonology": "diagnosis.dialog_pulmonology",
        "Gastroenterology": "diagnosis.dialog_gastroenterology",
        "Neurology": "diagnosis.dialog_neurology",
        "Endocrinology": "diagnosis.dialog_endocrinology",
        "Nephrology": "diagnosis.dialog_nephrology",
        "Rheumatology": "diagnosis.dialog_rheumatology",
        "Hematology": "diagnosis.dialog_hematology",
        "Dermatology": "diagnosis.dialog_dermatology",
    }

    async def execute(
        self,
        context: "WorkflowContext",
        dialog: str | None = None,
        prev_diagnosis: str | None = None,
        justification: str | None = None,
        specialty: str | None = None,
        previous_predictions: list[dict] | None = None,
        critique: str | None = None,
        **kwargs
    ) -> DiagnosisDialogAgentOutput:
        """Make a diagnosis from the dialog.
        
        Args:
            context: Workflow context containing dialog history
            dialog: Optional dialog string override
            prev_diagnosis: Previous diagnosis to avoid (if re-diagnosing)
            justification: Reason the previous diagnosis was rejected
            specialty: Medical specialty for specialized prompts (None or "Others" uses default)
            previous_predictions: Previous round's predictions for anchored diagnosis mode
            
        Returns:
            DiagnosisDialogAgentOutput with disease, confidence, reasoning
        """
        # Get dialog from context or parameter
        if dialog:
            dialog_str = dialog
        elif context.dialog and context.dialog.round_count() > 0:
            dialog_str = context.dialog.format_readable()
        else:
            dialog_str = ""
        
        if not dialog_str or dialog_str.strip() == "":
            return DiagnosisDialogAgentOutput(
                success=False,
                error="No dialog to analyze",
                predictions=[]
            )
        
        # Determine which prompt to use based on config flags
        reasoning_first = self.config.get("reasoning_first", False)

        prompt_agent_name = self.agent_name
        if reasoning_first:
            prompt_agent_name = "diagnosis.dialog_rf"
        if specialty and specialty != "Others" and specialty in self.SPECIALTY_AGENT_MAP:
            prompt_agent_name = self.SPECIALTY_AGENT_MAP[specialty]

        if prev_diagnosis:
            user_prompt = self._prompt_loader.format_user_prompt(
                prompt_agent_name,
                "user_prompt_template_retry",
                dialog=dialog_str,
                prev_diagnosis=prev_diagnosis,
                justification=justification or "Not justified by the available information."
            )
        else:
            user_prompt = self._prompt_loader.format_user_prompt(
                prompt_agent_name,
                "user_prompt_template",
                dialog=dialog_str
            )
        
        # Get specialty-specific system prompt if applicable
        if prompt_agent_name != self.agent_name:
            system_prompt = self._prompt_loader.get_system_prompt(prompt_agent_name)
        else:
            system_prompt = None

        try:
            response_model = DiagnosisDialogResponseReasoningFirst if reasoning_first else DiagnosisDialogResponse
            result = await self._call_llm_structured(user_prompt, response_model, system_prompt=system_prompt)
            
            # Convert reasoning-first predictions to standard format for downstream compatibility
            if reasoning_first:
                predictions = [
                    DiagnosisPrediction(disease=p.disease, confidence=p.confidence, reasoning=p.reasoning)
                    for p in result.predictions
                ]
            else:
                predictions = result.predictions
            
            # Sort by confidence descending
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            
            if not predictions:
                return DiagnosisDialogAgentOutput(
                    success=False,
                    error="LLM returned empty predictions",
                    predictions=[]
                )
            
            return DiagnosisDialogAgentOutput(
                success=True,
                predictions=predictions
            )
        except Exception as e:
            print(f"  [LLM_ERROR] diagnosis_dialog_agent failed: {str(e)}")
            return DiagnosisDialogAgentOutput(
                success=False,
                error=str(e),
                predictions=[]
            )
    
    def to_diagnosis_result(self, output: DiagnosisDialogAgentOutput) -> DiagnosisResult:
        """Convert output to DiagnosisResult model."""
        return DiagnosisResult(
            disease=output.disease,
            confidence=output.confidence,
            reasoning=output.reasoning
        )
