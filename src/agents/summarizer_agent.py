"""Summarizer Agent - Creates a summary of the doctor-patient dialog."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.llm.llm_client import StructuredOutputError
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class SummaryResponse(BaseModel):
    """Structured response for case summary."""
    summary: str = Field(..., description="Concise case summary of the patient from the dialog")


class SummarizerAgentOutput(AgentOutput):
    """Output from the summarizer agent."""
    summary: str = Field(default="", description="Summary of the patient's case from the dialog")


class SummarizerAgent(BaseAgent):
    """Agent that summarizes the doctor-patient dialog into a case summary.
    
    The summarizer agent:
    - Takes the dialog in 'D': <q>, 'P': <ans> format
    - Creates a concise case summary suitable for diagnosis
    - Only includes information explicitly provided by the patient
    - Does not add any extra information or inferences
    """
    
    agent_name = "summarizer.default"

    async def execute(
        self,
        context: "WorkflowContext",
        dialog: str | None = None,
        **kwargs
    ) -> SummarizerAgentOutput:
        """Generate a case summary from the dialog.
        
        Args:
            context: Workflow context containing dialog history
            dialog: Optional dialog string override
            
        Returns:
            SummarizerAgentOutput with the case summary
        """
        # Get dialog from context or parameter
        if dialog:
            dialog_str = dialog
        elif context.dialog and context.dialog.round_count() > 0:
            dialog_str = context.dialog.format_readable()
        else:
            dialog_str = ""
        
        if not dialog_str or dialog_str.strip() == "":
            return SummarizerAgentOutput(
                success=False,
                error="No dialog to summarize",
                summary=""
            )
        
        user_prompt = self.get_user_prompt(
            "user_prompt_template",
            dialog=dialog_str
        )

        try:
            result = await self._call_llm_structured(user_prompt, SummaryResponse)
            summary = result.summary.strip()

            return SummarizerAgentOutput(
                success=True,
                summary=summary
            )
        except StructuredOutputError as e:
            # The raw response IS the summary, just not JSON-wrapped.
            # Use it directly instead of failing.
            raw = e.raw_response.strip()
            print(f"  [SUMMARIZER] Structured parse failed, using raw response as summary (length={len(raw)})")
            print(f"  [SUMMARIZER] Raw output preview: {raw}...")
            return SummarizerAgentOutput(
                success=True,
                summary=raw
            )
        except Exception as e:
            print(f"  [LLM_ERROR] summarizer_agent failed: {str(e)}")
            return SummarizerAgentOutput(
                success=False,
                error=str(e),
                summary=""
            )
