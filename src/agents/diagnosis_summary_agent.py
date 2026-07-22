"""Doctor Diagnosis Summary Agent - Makes diagnosis from case summary."""

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


class DiagnosisSummaryResponse(BaseModel):
    """Structured response for diagnosis predictions."""
    predictions: list[DiagnosisPrediction] = Field(..., description="Top 3 disease predictions sorted by confidence")


class DiagnosisSummaryAgentOutput(AgentOutput):
    """Output from the diagnosis summary agent."""
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


class DiagnosisSummaryAgent(BaseAgent):
    """Agent that diagnoses disease based on a case summary.

    The diagnosis agent:
    - Receives a summary of the doctor-patient dialog (not the raw dialog)
    - Analyzes the summary to make a diagnosis
    - Outputs the disease name and confidence level (0-100)
    - Confidence >= the workflow's ``high_confidence_threshold`` triggers the
      early-exit condition in the workflow loop
    - Can receive previous diagnosis to avoid repetition
    """
    
    agent_name = "diagnosis.summary"
    
    @property
    def _prompt_key(self) -> str:
        """Return prompt key based on summary_type config."""
        if self.config.get("summary_type") == "default":
            return "diagnosis.summary_para"
        return "diagnosis.summary"
    
    # Mapping from specialty to agent name for specialized prompts
    SPECIALTY_AGENT_MAP = {
        "Cardiology": "diagnosis.summary_cardiology",
        "Pulmonology": "diagnosis.summary_pulmonology",
        "Gastroenterology": "diagnosis.summary_gastroenterology",
        "Neurology": "diagnosis.summary_neurology",
        "Endocrinology": "diagnosis.summary_endocrinology",
        "Nephrology": "diagnosis.summary_nephrology",
        "Rheumatology": "diagnosis.summary_rheumatology",
        "Hematology": "diagnosis.summary_hematology",
        "Dermatology": "diagnosis.summary_dermatology",
    }

    async def execute(
        self,
        context: "WorkflowContext",
        summary: str | None = None,
        prev_diagnosis: str | None = None,
        justification: str | None = None,
        specialty: str | None = None,
        **kwargs
    ) -> DiagnosisSummaryAgentOutput:
        """Make a diagnosis from the summary.
        
        Args:
            context: Workflow context containing summary
            summary: Optional summary string override
            prev_diagnosis: Previous diagnosis to avoid (if re-diagnosing)
            justification: Reason the previous diagnosis was rejected
            specialty: Medical specialty for specialized prompts (None or "Others" uses default)
            
        Returns:
            DiagnosisSummaryAgentOutput with disease, confidence, reasoning
        """
        # Get summary from context or parameter
        summary_str = summary or context.get("summary", "")
        
        if not summary_str or summary_str.strip() == "":
            return DiagnosisSummaryAgentOutput(
                success=False,
                error="No summary to analyze",
                predictions=[]
            )
        
        # Determine which agent name to use for prompts
        prompt_agent_name = self._prompt_key
        if specialty and specialty != "Others" and specialty in self.SPECIALTY_AGENT_MAP:
            prompt_agent_name = self.SPECIALTY_AGENT_MAP[specialty]
            # print(f"  [SPECIALTY] Using {specialty} specialist prompt")
        
        # Get user prompt template
        if prev_diagnosis:
            user_prompt = self._prompt_loader.format_user_prompt(
                prompt_agent_name,
                "user_prompt_template_retry",
                summary=summary_str,
                prev_diagnosis=prev_diagnosis,
                justification=justification or "Not justified by the available information."
            )
        else:
            user_prompt = self._prompt_loader.format_user_prompt(
                prompt_agent_name,
                "user_prompt_template",
                summary=summary_str
            )
        
        # Get specialty-specific system prompt if applicable
        if prompt_agent_name != self.agent_name:
            system_prompt = self._prompt_loader.get_system_prompt(prompt_agent_name)
        else:
            system_prompt = None  # Will use default from base class

        try:
            result = await self._call_llm_structured(user_prompt, DiagnosisSummaryResponse, system_prompt=system_prompt)
            
            predictions = result.predictions
            
            # Sort by confidence descending
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            
            if not predictions:
                return DiagnosisSummaryAgentOutput(
                    success=False,
                    error=f"LLM returned empty predictions. Raw result: {str(result)[:500]}",
                    predictions=[]
                )
            
            return DiagnosisSummaryAgentOutput(
                success=True,
                predictions=predictions
            )
        except Exception as e:
            print(f"  [LLM_ERROR] diagnosis_summary_agent failed: {str(e)}")
            return DiagnosisSummaryAgentOutput(
                success=False,
                error=str(e),
                predictions=[]
            )
    
    def to_diagnosis_result(self, output: DiagnosisSummaryAgentOutput) -> DiagnosisResult:
        """Convert output to DiagnosisResult model."""
        return DiagnosisResult(
            disease=output.disease,
            confidence=output.confidence,
            reasoning=output.reasoning
        )
