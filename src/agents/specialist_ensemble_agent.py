"""Specialist Ensemble Agent - Merges diagnoses from multiple specialist physicians."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.agents.diagnosis_dialog_agent import DiagnosisPrediction, DiagnosisDialogResponse, DiagnosisDialogAgentOutput
from src.llm.llm_client import StructuredOutputError
from pydantic import Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


SPECIALTIES = [
    "Cardiology",
    "Pulmonology",
    "Gastroenterology",
    "Neurology",
    "Endocrinology",
    "Nephrology",
    "Rheumatology",
    "Hematology",
    "Dermatology",
]


class SpecialistEnsembleAgent(BaseAgent):
    """Agent that merges diagnoses from multiple specialist physicians.
    
    Takes predictions from 9 specialists + 1 generalist (30 total predictions)
    and produces a final merged set of 3 predictions.
    """
    
    agent_name = "ensemble.specialist_merge"

    async def execute(
        self,
        context: "WorkflowContext",
        specialist_predictions: dict[str, list[dict]] | None = None,
        dialog: str | None = None,
        **kwargs
    ) -> DiagnosisDialogAgentOutput:
        """Merge specialist predictions into final diagnosis.
        
        Args:
            context: Workflow context
            specialist_predictions: Dict mapping specialty name -> list of prediction dicts
                                   e.g. {"Cardiology": [{"disease": ..., "confidence": ..., "reasoning": ...}], ...}
            dialog: Full dialog string
            
        Returns:
            DiagnosisDialogAgentOutput with merged predictions
        """
        if not specialist_predictions:
            return DiagnosisDialogAgentOutput(
                success=False,
                error="No specialist predictions to merge",
                predictions=[]
            )
        
        # Format specialist predictions for the prompt
        formatted_parts = []
        for specialty, preds in specialist_predictions.items():
            formatted_parts.append(f"\n{specialty}:")
            for i, p in enumerate(preds[:3], 1):
                disease = p.get("disease", "Unknown")
                confidence = p.get("confidence", 0)
                reasoning = p.get("reasoning", "")
                formatted_parts.append(f"  {i}. {disease} (confidence: {confidence}%)")
                if reasoning:
                    formatted_parts.append(f"     Reasoning: {reasoning}")
        specialist_predictions_str = "\n".join(formatted_parts)
        
        # Get dialog
        if dialog is None:
            dialog = context.get_dialog_readable() if hasattr(context, 'get_dialog_readable') else ""
        
        user_prompt = self.get_user_prompt(
            "user_prompt_template",
            dialog=dialog,
            specialist_predictions=specialist_predictions_str
        )
        
        try:
            result = await self._call_llm_structured(user_prompt, DiagnosisDialogResponse)
            
            return DiagnosisDialogAgentOutput(
                success=True,
                predictions=result.predictions
            )
        except StructuredOutputError as e:
            print(f"  [FALLBACK] specialist_ensemble_agent structured output failed: {str(e)[:100]}")
            return DiagnosisDialogAgentOutput(
                success=False,
                error=f"Structured output failed: {str(e)[:100]}",
                predictions=[]
            )
        except Exception as e:
            print(f"  [LLM_ERROR] specialist_ensemble_agent failed: {str(e)}")
            return DiagnosisDialogAgentOutput(
                success=False,
                error=str(e),
                predictions=[]
            )
