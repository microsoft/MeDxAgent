"""Doctor Question Evidence Agent - Generates questions guided by general evidence gaps."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.llm.llm_client import StructuredOutputError
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class DoctorEvidenceQuestionResponse(BaseModel):
    """Structured response for evidence-guided question generation."""
    question: str = Field(..., description="Your specific diagnostic question - ONE clear atomic question guided by the evidence gap report")


class DoctorQEvidenceAgentOutput(AgentOutput):
    """Output from the doctor question evidence agent."""
    question: str = Field(default="", description="ONE specific diagnostic question guided by evidence gaps")


class DoctorQEvidenceAgent(BaseAgent):
    """Agent that generates targeted questions using general evidence gap analysis.
    
    Unlike doctor_q_diff_evidence_agent which uses diagnosis-specific gaps,
    this agent uses general evidence gaps (missing patient information) to
    guide questioning during early rounds before differential diagnosis begins.
    """
    
    agent_name = "doctor.q_evidence"

    @property
    def _prompt_key(self) -> str:
        return "doctor.q_evidence"

    @property
    def system_prompt(self) -> str:
        return self._prompt_loader.get_system_prompt(self._prompt_key)

    async def execute(
        self,
        context: "WorkflowContext",
        evidence_gap_report: str | None = None,
        current_round: int | None = None,
        max_rounds: int | None = None,
        **kwargs
    ) -> DoctorQEvidenceAgentOutput:
        """Generate the next diagnostic question using general evidence gap analysis.
        
        Args:
            context: Workflow context containing dialog history
            evidence_gap_report: Formatted general evidence gap report
            current_round: Current round number (1-indexed)
            max_rounds: Maximum number of rounds allowed
            
        Returns:
            DoctorQEvidenceAgentOutput with the question
        """
        # Get explicit list of previously asked questions
        asked_questions = context.get_doctor_questions()
        if asked_questions:
            asked_questions_str = "\n".join(f"{i}. {q}" for i, q in enumerate(asked_questions, 1))
        else:
            asked_questions_str = "(No questions asked yet)"
        
        # Evidence gap report
        evidence_gap_str = evidence_gap_report or "(No evidence gap report available)"
        
        # Dialog
        dialog_str = context.get_dialog_readable() if hasattr(context, 'get_dialog_readable') else ""
        
        # Format round information
        if current_round is not None and max_rounds is not None:
            rounds_remaining = max_rounds - current_round
            round_info = f"Round {current_round} of {max_rounds} ({rounds_remaining} questions remaining after this one)"
        else:
            round_info = "(Round information not available)"
        
        user_prompt = self._prompt_loader.format_user_prompt(
            self._prompt_key,
            "user_prompt_template",
            evidence_gap_report=evidence_gap_str,
            asked_questions=asked_questions_str,
            dialog=dialog_str,
            round_info=round_info
        )

        try:
            result = await self._call_llm_structured(user_prompt, DoctorEvidenceQuestionResponse)
            question = result.question
            
            # Clean up question
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            if question.startswith("'") and question.endswith("'"):
                question = question[1:-1]
            
            question = question.strip()
            if not question:
                return DoctorQEvidenceAgentOutput(
                    success=False,
                    error="LLM returned empty question",
                    question=""
                )
            
            return DoctorQEvidenceAgentOutput(
                success=True,
                question=question
            )
        except StructuredOutputError as e:
            # Fallback: use raw response directly as the question
            raw = str(e.raw_response).strip() if hasattr(e, 'raw_response') else ""
            if raw and len(raw) < 500:
                return DoctorQEvidenceAgentOutput(
                    success=True,
                    question=raw
                )
            return DoctorQEvidenceAgentOutput(
                success=False,
                error=f"Structured output parsing failed: {str(e)}",
                question=""
            )
        except Exception as e:
            print(f"  [LLM_ERROR] doctor_q_evidence_agent failed: {str(e)}")
            return DoctorQEvidenceAgentOutput(
                success=False,
                error=str(e),
                question=""
            )
