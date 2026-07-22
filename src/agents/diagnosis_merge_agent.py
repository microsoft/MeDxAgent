"""Diagnosis Merge Agent - Merges KB and LLM diagnoses into final predictions."""

from src.agents.base_agent import BaseAgent, AgentOutput
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class MergedDiagnosisPrediction(BaseModel):
    """A merged diagnosis prediction with confidence."""
    disease: str = Field(..., description="Final predicted disease name")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    reasoning: str = Field(default="", description="Combined reasoning for this diagnosis")


class DiagnosisMergeResponse(BaseModel):
    """Structured response for merged diagnosis."""
    predictions: list[MergedDiagnosisPrediction] = Field(
        ..., 
        description="Top 3 final diagnoses sorted by confidence, based on your independent clinical evaluation of the evidence"
    )


class DiagnosisMergeAgentOutput(AgentOutput):
    """Output from the diagnosis merge agent."""
    predictions: list[MergedDiagnosisPrediction] = Field(
        default_factory=list, 
        description="Top 3 merged disease predictions"
    )
    
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


class DiagnosisMergeAgent(BaseAgent):
    """Agent that merges diagnoses from KB and LLM sources.
    
    The diagnosis merge agent:
    - Receives structured patient summary
    - Receives 3 diagnoses + reasoning from diagnosis_kb_agent (KB-grounded)
    - Receives 3 diagnoses + reasoning from diagnosis_summary_agent (LLM reasoning)
    - Synthesizes both inputs considering agreement/disagreement
    - Outputs TOP 3 final diagnoses with NEW confidence scores
    - Confidence reflects certainty after considering both sources
    """
    
    agent_name = "diagnosis.merge"

    @property
    def _prompt_key(self) -> str:
        """Return prompt key based on config."""
        if self.config.get("use_paragraph_summary", False):
            return "diagnosis.merge_para"
        if self.config.get("use_dialog_for_merge", False):
            return "diagnosis.merge_dialog"
        return "diagnosis.merge"

    @property
    def system_prompt(self) -> str:
        return self._prompt_loader.get_system_prompt(self._prompt_key)

    async def execute(
        self,
        context: "WorkflowContext",
        str_summary: dict | str | None = None,
        dialog: str | None = None,
        kb_predictions: list[dict] | None = None,
        llm_predictions: list[dict] | None = None,
        **kwargs
    ) -> DiagnosisMergeAgentOutput:
        """Merge KB and LLM diagnoses.
        
        Args:
            context: Workflow context
            str_summary: Structured patient summary
            kb_predictions: List of diagnoses from diagnosis_kb_agent
                           Each dict has 'disease' and 'reasoning'
            llm_predictions: List of diagnoses from diagnosis_summary_agent
                            Each dict has 'disease' and 'reasoning'
            
        Returns:
            DiagnosisMergeAgentOutput with top 3 merged predictions
        """
        # Get str_summary from context if not provided
        if str_summary is None:
            str_summary = context.get("str_summary", {})
        
        # Format summary for prompt
        if isinstance(str_summary, dict):
            import json
            summary_str = json.dumps(str_summary, indent=2)
        else:
            summary_str = str(str_summary) if str_summary else "{}"
        
        # Format KB predictions
        kb_str = self._format_predictions(kb_predictions, "Knowledge Base")
        
        # Format LLM predictions
        llm_str = self._format_predictions(llm_predictions, "Clinical Reasoning")
        
        # If both are empty, we can't merge
        if not kb_predictions and not llm_predictions:
            return DiagnosisMergeAgentOutput(
                success=False,
                error="No predictions from either KB or LLM to merge",
                predictions=[]
            )
        
        # Choose prompt based on dialog vs summary mode
        use_dialog_mode = self.config.get("use_dialog_for_merge", False)
        if use_dialog_mode:
            dialog_str = dialog or (context.get_dialog_readable() if hasattr(context, 'get_dialog_readable') else "")
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template",
                dialog=dialog_str,
                kb_predictions=kb_str,
                llm_predictions=llm_str
            )
        else:
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template",
                str_summary=summary_str,
                kb_predictions=kb_str,
                llm_predictions=llm_str
            )

        try:
            result = await self._call_llm_structured(user_prompt, DiagnosisMergeResponse)
            
            predictions = result.predictions[:3]  # Ensure max 3
            
            # Sort by confidence descending
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            
            if not predictions:
                return DiagnosisMergeAgentOutput(
                    success=False,
                    error="Merge produced no valid diagnoses",
                    predictions=[]
                )
            
            return DiagnosisMergeAgentOutput(
                success=True,
                predictions=predictions
            )
        except Exception as e:
            print(f"  [LLM_ERROR] diagnosis_merge_agent failed: {str(e)}")
            return DiagnosisMergeAgentOutput(
                success=False,
                error=str(e),
                predictions=[]
            )
    
    def _format_predictions(self, predictions: list[dict] | None, source: str) -> str:
        """Format predictions into a readable string."""
        if not predictions:
            return f"No predictions from {source}"
        
        lines = [f"From {source}:"]
        for i, pred in enumerate(predictions[:3], 1):
            disease = pred.get("disease", "Unknown")
            reasoning = pred.get("reasoning", "No reasoning provided")
            lines.append(f"{i}. {disease}")
            lines.append(f"   Reasoning: {reasoning}")
        return "\n".join(lines)
