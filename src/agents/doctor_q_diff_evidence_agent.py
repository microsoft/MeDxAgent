"""Doctor Question Differential Evidence Agent - Generates questions using evidence gap analysis."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.llm.llm_client import StructuredOutputError
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class DoctorDiffEvidenceQuestionResponse(BaseModel):
    """Structured response for evidence-guided diagnostic question generation."""
    question: str = Field(..., description="Your specific diagnostic question - ONE clear atomic question guided by the evidence gap analysis")


class DoctorQDiffEvidenceAgentOutput(AgentOutput):
    """Output from the doctor question differential evidence agent."""
    question: str = Field(default="", description="ONE specific diagnostic question guided by evidence gap analysis")


class DoctorQDiffEvidenceAgent(BaseAgent):
    """Agent that generates targeted questions using evidence gap analysis.
    
    The doctor question differential evidence agent:
    - Takes the top 3 differential diagnoses
    - Takes the evidence gap report identifying missing information
    - Asks ONE specific question that addresses the most critical evidence gap
    - Prioritizes questions that would most dramatically shift confidence between diagnoses
    """
    
    agent_name = "doctor.q_diff_evidence"

    @property
    def _prompt_key(self) -> str:
        """Return prompt key based on use_dialog_for_evidence_gap config."""
        if self.config.get("use_dialog_for_evidence_gap", False):
            return "doctor.q_diff_evidence_dialog"
        return "doctor.q_diff_evidence"

    @property
    def system_prompt(self) -> str:
        return self._prompt_loader.get_system_prompt(self._prompt_key)

    async def execute(
        self,
        context: "WorkflowContext",
        predictions: list[dict] | None = None,
        evidence_gap_report: str | None = None,
        str_summary: dict | str | None = None,
        current_round: int | None = None,
        max_rounds: int | None = None,
        **kwargs
    ) -> DoctorQDiffEvidenceAgentOutput:
        """Generate the next diagnostic question using evidence gap analysis.
        
        Args:
            context: Workflow context containing dialog history
            predictions: List of top 3 diagnoses with confidence scores
            evidence_gap_report: Formatted evidence gap report from evidence_gap_agent
            str_summary: Structured summary from summarizer_str_agent
            current_round: Current round number (1-indexed)
            max_rounds: Maximum number of rounds allowed
            
        Returns:
            DoctorQDiffEvidenceAgentOutput with the question
        """
        
        # Get explicit list of previously asked questions
        asked_questions = context.get_doctor_questions()
        if asked_questions:
            asked_questions_str = "\n".join(f"{i}. {q}" for i, q in enumerate(asked_questions, 1))
        else:
            asked_questions_str = "(No questions asked yet)"
        
        # Format predictions
        if predictions:
            predictions_str = self._format_predictions(predictions)
        else:
            predictions_str = "(No predictions available)"
        
        # Get structured summary
        if str_summary is None:
            str_summary = context.get("summary", "")
        
        if isinstance(str_summary, dict):
            import json
            str_summary_str = json.dumps(str_summary, indent=2)
        else:
            str_summary_str = str(str_summary) if str_summary else ""
        
        # Evidence gap report
        evidence_gap_str = evidence_gap_report or "(No evidence gap report available)"
        
        # Format round information
        if current_round is not None and max_rounds is not None:
            rounds_remaining = max_rounds - current_round
            round_info = f"Round {current_round} of {max_rounds} ({rounds_remaining} questions remaining after this one)"
        else:
            round_info = "(Round information not available)"
        
        # Choose prompt based on dialog vs summary mode
        use_dialog_mode = self.config.get("use_dialog_for_evidence_gap", False)
        if use_dialog_mode:
            dialog_str = context.get_dialog_readable() if hasattr(context, 'get_dialog_readable') else ""
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template",
                predictions=predictions_str,
                evidence_gap_report=evidence_gap_str,
                dialog=dialog_str,
                asked_questions=asked_questions_str,
                round_info=round_info
            )
        else:
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template",
                predictions=predictions_str,
                evidence_gap_report=evidence_gap_str,
                str_summary=str_summary_str,
                asked_questions=asked_questions_str,
                round_info=round_info
            )

        try:
            result = await self._call_llm_structured(user_prompt, DoctorDiffEvidenceQuestionResponse)
            question = result.question
            
            # Clean up question
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            if question.startswith("'") and question.endswith("'"):
                question = question[1:-1]
            
            question = question.strip()
            if not question:
                return DoctorQDiffEvidenceAgentOutput(
                    success=False,
                    error="LLM returned empty question",
                    question=""
                )
            
            return DoctorQDiffEvidenceAgentOutput(
                success=True,
                question=question
            )
        except StructuredOutputError as e:
            # Fallback: use raw response directly as the question
            raw_question = e.raw_response.strip()
            if raw_question.startswith('"') and raw_question.endswith('"'):
                raw_question = raw_question[1:-1]
            if raw_question.startswith("'") and raw_question.endswith("'"):
                raw_question = raw_question[1:-1]
            if raw_question:
                print(f"  [FALLBACK] Using raw response as diff evidence doctor question: {raw_question[:100]}...")
                return DoctorQDiffEvidenceAgentOutput(
                    success=True,
                    question=raw_question
                )
            return DoctorQDiffEvidenceAgentOutput(
                success=False,
                error=str(e),
                question=""
            )
        except Exception as e:
            print(f"  [LLM_ERROR] doctor_q_diff_evidence_agent failed: {str(e)}")
            return DoctorQDiffEvidenceAgentOutput(
                success=False,
                error=str(e),
                question=""
            )
    
    def _format_predictions(self, predictions: list[dict]) -> str:
        """Format predictions list into a readable string."""
        lines = []
        for i, pred in enumerate(predictions[:5], 1):
            disease = pred.get("disease", "Unknown")
            confidence = pred.get("confidence", 0)
            reasoning = pred.get("reasoning", "")
            lines.append(f"{i}. {disease} (confidence: {confidence}%)")
            if reasoning:
                lines.append(f"   Reasoning: {reasoning}")
        return "\n".join(lines)
