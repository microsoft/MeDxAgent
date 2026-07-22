"""Diagnosis KB Agent - Makes diagnosis from knowledge graph disease candidates."""

from src.agents.base_agent import BaseAgent, AgentOutput
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class KBDiagnosisPrediction(BaseModel):
    """A single diagnosis prediction from KB analysis."""
    disease: str = Field(..., description="Disease name selected from the KB candidates")
    reasoning: str = Field(default="", description="Key clinical findings and test results supporting this diagnosis")


class DiagnosisKBResponse(BaseModel):
    """Structured response for KB-based diagnosis."""
    predictions: list[KBDiagnosisPrediction] = Field(
        ..., 
        description="Top 3 disease predictions selected from KB candidates, ranked by likelihood based on clinical plausibility not just symptom match count"
    )


class DiagnosisKBAgentOutput(AgentOutput):
    """Output from the diagnosis KB agent."""
    predictions: list[KBDiagnosisPrediction] = Field(
        default_factory=list, 
        description="Top 3 disease predictions from KB candidates, ranked by clinical plausibility"
    )
    
    @property
    def disease(self) -> str:
        """Get the top predicted disease."""
        return self.predictions[0].disease if self.predictions else ""
    
    @property
    def reasoning(self) -> str:
        """Get the top prediction's reasoning."""
        return self.predictions[0].reasoning if self.predictions else ""


class DiagnosisKBAgent(BaseAgent):
    """Agent that selects diagnoses from knowledge graph candidates.
    
    The diagnosis KB agent:
    - Receives a structured patient summary and a list of disease candidates from UMLS
    - Critically analyzes which KB candidates match the patient's presentation
    - Filters out non-disease entries (relationships, symptoms, etc.)
    - Selects the TOP 3 most likely diseases from the KB list
    - Provides reasoning for each selection based on patient evidence
    """
    
    agent_name = "diagnosis.kb"

    @property
    def _prompt_key(self) -> str:
        if self.config.get("use_paragraph_summary", False):
            return "diagnosis.kb_para"
        if self.config.get("use_dialog_for_kb", False):
            return "diagnosis.kb_dialog"
        return "diagnosis.kb"

    @property
    def system_prompt(self) -> str:
        return self._prompt_loader.get_system_prompt(self._prompt_key)

    async def execute(
        self,
        context: "WorkflowContext",
        str_summary: dict | str | None = None,
        dialog: str | None = None,
        kb_candidates: str | None = None,
        **kwargs
    ) -> DiagnosisKBAgentOutput:
        """Select diagnoses from KB candidates based on patient summary.
        
        Args:
            context: Workflow context
            str_summary: Structured patient summary from summarizer_str_agent
            kb_candidates: Formatted string of disease candidates from UMLS
            
        Returns:
            DiagnosisKBAgentOutput with top 3 disease predictions
        """
        # Get str_summary from context if not provided
        # print("[diagnosis_kb_agent] execute called")
        use_dialog_mode = self.config.get("use_dialog_for_kb", False)
        
        if use_dialog_mode:
            dialog_str = dialog or context.get_dialog_readable()
        else:
            if str_summary is None:
                str_summary = context.get("str_summary", {})
            
            # Format summary for prompt
            if isinstance(str_summary, dict):
                import json
                summary_str = json.dumps(str_summary, indent=2)
            else:
                summary_str = str(str_summary) if str_summary else "{}"
        
        # Check if we have KB candidates
        if not kb_candidates:
            return DiagnosisKBAgentOutput(
                success=False,
                error="No KB candidates provided",
                predictions=[]
            )
        
        if use_dialog_mode:
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template",
                dialog=dialog_str,
                kb_candidates=kb_candidates
            )
        else:
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template",
                str_summary=summary_str,
                kb_candidates=kb_candidates
            )

        try:

            # print("\n[INPUT] DIAGNOSIS KB\n")
            # print(user_prompt)
            # print("\n[END INPUT] DIAGNOSIS KB\n")
            result = await self._call_llm_structured(user_prompt, DiagnosisKBResponse)
            # print("\n[OUTPUT] DIAGNOSIS KB\n")
            # print(result.predictions)
            # print("\n[OUTPUT] DIAGNOSIS KB\n")
            
            predictions = result.predictions[:3]  # Ensure max 3
            
            if not predictions:
                return DiagnosisKBAgentOutput(
                    success=False,
                    error="No valid diagnoses selected from KB candidates",
                    predictions=[]
                )
            
            return DiagnosisKBAgentOutput(
                success=True,
                predictions=predictions
            )
        except Exception as e:
            print(f"  [LLM_ERROR] diagnosis_kb_agent failed: {str(e)}")
            return DiagnosisKBAgentOutput(
                success=False,
                error=str(e),
                predictions=[]
            )
