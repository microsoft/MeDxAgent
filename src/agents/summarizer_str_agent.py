"""Structured Summarizer Agent - Creates and updates a structured diagnostic state."""

from src.agents.base_agent import BaseAgent, AgentOutput
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class StructuredSummary(BaseModel):
    """Structured diagnostic state for a patient."""
    demographics: str = Field(default="", description="Age, gender, occupation (e.g., '45-year-old female, nurse')")
    vital_signs: list[str] = Field(default_factory=list, description="BP, HR, temp, O2 sat, RR, etc. with exact values (e.g., 'BP 230/150 mmHg', 'HR 110 bpm')")
    symptoms_present: list[str] = Field(default_factory=list, description="Simple Canonical symptom names ONLY - use standard medical terms without qualifiers (e.g., 'shortness of breath', 'chest pain', 'fever', 'cough', 'headache'). Do NOT include severity, timing, or triggers here.")
    symptoms_absent: list[str] = Field(default_factory=list, description="Symptoms the patient explicitly denies (simple names only)")
    symptom_details: list[str] = Field(default_factory=list, description="Qualitative details about symptoms including severity, character, timing, location, triggers, and relieving factors (e.g., 'shortness of breath: mild, exertional, relieved by rest', 'chest pain: sharp, substernal, radiates to left arm, worse with exertion')")
    onset: str = Field(default="", description="When symptoms started and their duration")
    medical_history: list[str] = Field(default_factory=list, description="Past illnesses, surgeries, hospitalizations")
    medications: list[str] = Field(default_factory=list, description="Current and recent medications with doses if provided and their outcomes")
    physical_exam: list[str] = Field(default_factory=list, description="Examination findings (e.g., 'bilateral crackles in lungs', 'hepatomegaly')")
    risk_factors: list[str] = Field(default_factory=list, description="Predisposing conditions, lifestyle factors, family history")
    recent_exposures: list[str] = Field(default_factory=list, description="Sick contacts, food exposures, environmental exposures, recent travel")
    tests: list[str] = Field(default_factory=list, description="ALL test results including lab values, imaging, ECG, biopsies, etc. with EXACT numerical values")


class SummarizerStrAgentOutput(AgentOutput):
    """Output from the structured summarizer agent."""
    str_summary: StructuredSummary = Field(
        default_factory=StructuredSummary,
        description="Structured diagnostic state of the patient"
    )


class SummarizerStrAgent(BaseAgent):
    """Agent that creates and updates a structured diagnostic state.
    
    The structured summarizer agent:
    - Takes the last conversational turn of doctor-patient dialog
    - Takes the structured summary so far (can be null initially)
    - Updates the structured summary with new information from the last turn
    - Only includes information explicitly provided by the patient in answers
    - Does not add any extra information or inferences
    """
    
    agent_name = "summarizer.structured"

    async def execute(
        self,
        context: "WorkflowContext",
        dialog_lastturn: str | None = None,
        str_summary: dict | StructuredSummary | None = None,
        **kwargs
    ) -> SummarizerStrAgentOutput:
        """Update the structured summary with information from the last dialog turn.
        
        Args:
            context: Workflow context
            dialog_lastturn: The last turn of dialog in 'D': <q>, 'P': <ans> format
            str_summary: The structured summary so far (can be None to start)
            
        Returns:
            StrSummarizerAgentOutput with the updated structured summary
        """
        # Get dialog_lastturn from context if not provided
        if not dialog_lastturn:
            if context.dialog and context.dialog.round_count() > 0:
                last_turn = context.dialog.turns[-1]
                dialog_lastturn = f"D: {last_turn.question}\nP: {last_turn.answer}"
            else:
                return SummarizerStrAgentOutput(
                    success=False,
                    error="No dialog turn to process",
                    str_summary=StructuredSummary()
                )
        
        # Convert str_summary to JSON string for the prompt
        if str_summary is None:
            summary_json = '{"demographics": "", "vital_signs": [], "symptoms_present": [], "symptoms_absent": [], "symptom_details": [], "onset": "", "medical_history": [], "medications": [], "physical_exam": [], "risk_factors": [], "recent_exposures": [], "tests": []}'
        elif isinstance(str_summary, StructuredSummary):
            summary_json = str_summary.model_dump_json()
        elif isinstance(str_summary, dict):
            import json
            summary_json = json.dumps(str_summary)
        else:
            summary_json = str(str_summary)
        
        user_prompt = self.get_user_prompt(
            "user_prompt_template",
            dialog_lastturn=dialog_lastturn,
            str_summary=summary_json
        )

        try:
            result = await self._call_llm_structured(user_prompt, StructuredSummary)
            
            return SummarizerStrAgentOutput(
                success=True,
                str_summary=result
            )
        except Exception as e:
            print(f"  [LLM_ERROR] summarizer_str_agent failed: {str(e)}")
            return SummarizerStrAgentOutput(
                success=False,
                error=f"{str(e)}. Check LLM response format.",
                str_summary=StructuredSummary()
            )
