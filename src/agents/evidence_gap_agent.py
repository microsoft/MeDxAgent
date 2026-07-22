"""Evidence Gap Agent - Analyzes what information is missing for diagnosis."""

from src.agents.base_agent import BaseAgent, AgentOutput
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class DiagnosisGap(BaseModel):
    """Evidence gaps for a specific diagnosis."""
    disease: str = Field(..., description="Disease name")
    missing_evidence: str = Field(default="", description="2-3 sentences describing all the tests, symptoms, findings, and history needed to confirm or rule out this diagnosis")


class EvidenceGapReport(BaseModel):
    """Structured report of evidence gaps."""
    general_gaps: list[str] = Field(default_factory=list, description="General patient information that is missing (demographics, vitals, history, etc.)")
    diagnosis_specific_gaps: list[DiagnosisGap] = Field(default_factory=list, description="Evidence gaps specific to each differential diagnosis")


class GeneralEvidenceGapReport(BaseModel):
    """Report of general evidence gaps only (no diagnosis-specific gaps)."""
    general_gaps: list[str] = Field(default_factory=list, description="General patient information that is missing")


class EvidenceGapAgentOutput(AgentOutput):
    """Output from the evidence gap agent."""
    report: EvidenceGapReport = Field(default_factory=EvidenceGapReport, description="The evidence gap report")
    
    def format_for_doctor(self) -> str:
        """Format the report as a string for the doctor agent."""
        lines = []
        
        if self.report.general_gaps:
            lines.append("GENERAL INFORMATION GAPS:")
            for gap in self.report.general_gaps:
                lines.append(f"  - {gap}")
            lines.append("")
        
        if self.report.diagnosis_specific_gaps:
            lines.append("DIAGNOSIS-SPECIFIC EVIDENCE NEEDED:")
            for diag in self.report.diagnosis_specific_gaps:
                lines.append(f"  {diag.disease}:")
                lines.append(f"    {diag.missing_evidence}")
            lines.append("")
        
        return "\n".join(lines)


class GeneralEvidenceGapAgentOutput(AgentOutput):
    """Output from the evidence gap agent in general mode."""
    report: GeneralEvidenceGapReport = Field(default_factory=GeneralEvidenceGapReport, description="The general evidence gap report")
    
    def format_for_doctor(self) -> str:
        """Format the general gap report as a string for the doctor agent."""
        lines = []
        if self.report.general_gaps:
            lines.append("MISSING PATIENT INFORMATION:")
            for gap in self.report.general_gaps:
                lines.append(f"  - {gap}")
            lines.append("")
        return "\n".join(lines)


class EvidenceGapAgent(BaseAgent):
    """Agent that analyzes what information is missing for diagnosis.
    
    The evidence gap agent:
    - Reviews the current structured summary to identify missing general information
    - Analyzes each differential diagnosis to identify what evidence would confirm/exclude it
    - Outputs a structured report that can guide the doctor's questioning
    """
    
    agent_name = "evidence_gap.default"

    @property
    def _prompt_key(self) -> str:
        """Return prompt key — always evidence_gap.default for execute()."""
        return "evidence_gap.default"

    @property
    def _user_prompt_template_key(self) -> str:
        """Return the user prompt template name based on config."""
        if self.config.get("use_paragraph_summary", False):
            return "user_prompt_template_para"
        if self.config.get("use_dialog_for_evidence_gap", False):
            return "user_prompt_template_dialog"
        return "user_prompt_template"

    @property
    def system_prompt(self) -> str:
        return self._prompt_loader.get_system_prompt(self._prompt_key)

    async def execute(
        self,
        context: "WorkflowContext",
        predictions: list[dict] | None = None,
        str_summary: dict | str | None = None,
        dialog: str | None = None,
        **kwargs
    ) -> EvidenceGapAgentOutput:
        """Analyze evidence gaps in the current clinical picture.
        
        Args:
            context: Workflow context
            predictions: List of top 3 diagnoses with disease, confidence, reasoning
            str_summary: Structured summary from summarizer_str_agent
            
        Returns:
            EvidenceGapAgentOutput with the evidence gap report
        """
        # Get structured summary
        if str_summary is None:
            str_summary = context.get("summary", "")
        
        if isinstance(str_summary, dict):
            import json
            str_summary_str = json.dumps(str_summary, indent=2)
        else:
            str_summary_str = str(str_summary) if str_summary else ""
        
        # Format predictions
        if predictions:
            predictions_str = self._format_predictions(predictions)
        else:
            predictions_str = "(No predictions available)"
        
        # Choose prompt based on dialog vs summary mode
        use_dialog_mode = self.config.get("use_dialog_for_evidence_gap", False)
        template_key = self._user_prompt_template_key
        
        if use_dialog_mode:
            dialog_str = dialog or (context.get_dialog_readable() if hasattr(context, 'get_dialog_readable') else "")
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                template_key,
                dialog=dialog_str,
                predictions=predictions_str
            )
        else:
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                template_key,
                str_summary=str_summary_str,
                predictions=predictions_str
            )

        try:
            result = await self._call_llm_structured(user_prompt, EvidenceGapReport)
            
            return EvidenceGapAgentOutput(
                success=True,
                report=result
            )
        except Exception as e:
            print(f"  [LLM_ERROR] evidence_gap_agent failed: {str(e)}")
            return EvidenceGapAgentOutput(
                success=False,
                error=str(e),
                report=EvidenceGapReport()
            )
    
    def _format_predictions(self, predictions: list[dict]) -> str:
        """Format predictions list into a readable string."""
        lines = []
        for i, pred in enumerate(predictions[:3], 1):
            disease = pred.get("disease", "Unknown")
            confidence = pred.get("confidence", 0)
            reasoning = pred.get("reasoning", "")
            lines.append(f"{i}. {disease} (confidence: {confidence}%)")
            if reasoning:
                lines.append(f"   Reasoning: {reasoning}")
        return "\n".join(lines)

    async def execute_general(
        self,
        context: "WorkflowContext",
        str_summary: dict | str | None = None,
        dialog: str | None = None,
        **kwargs
    ) -> GeneralEvidenceGapAgentOutput:
        """Analyze general evidence gaps (no predictions needed).
        
        Args:
            context: Workflow context
            str_summary: Structured summary from summarizer_str_agent
            dialog: Dialog string (used when use_dialog_for_evidence_gap is True)
            
        Returns:
            GeneralEvidenceGapAgentOutput with general gaps only
        """
        use_dialog_mode = self.config.get("use_dialog_for_evidence_gap", False)
        
        general_key = "evidence_gap.general"
        if self.config.get("use_paragraph_summary", False):
            general_template = "user_prompt_template_para"
        elif use_dialog_mode:
            general_template = "user_prompt_template_dialog"
        else:
            general_template = "user_prompt_template"
        
        if use_dialog_mode:
            dialog_str = dialog or (context.get_dialog_readable() if hasattr(context, 'get_dialog_readable') else "")
            user_prompt = self._prompt_loader.format_user_prompt(
                general_key,
                general_template,
                dialog=dialog_str
            )
        else:
            if str_summary is None:
                str_summary = context.get("summary", "")
            
            if isinstance(str_summary, dict):
                import json
                str_summary_str = json.dumps(str_summary, indent=2)
            else:
                str_summary_str = str(str_summary) if str_summary else ""
            
            user_prompt = self._prompt_loader.format_user_prompt(
                general_key,
                general_template,
                str_summary=str_summary_str
            )

        try:
            result = await self._call_llm_structured(
                user_prompt, GeneralEvidenceGapReport,
                system_prompt=self._prompt_loader.get_system_prompt(general_key)
            )
            
            return GeneralEvidenceGapAgentOutput(
                success=True,
                report=result
            )
        except Exception as e:
            print(f"  [LLM_ERROR] evidence_gap_agent (general) failed: {str(e)}")
            return GeneralEvidenceGapAgentOutput(
                success=False,
                error=str(e),
                report=GeneralEvidenceGapReport()
            )
