"""Doctor Question Differential Agent - Generates targeted questions to differentiate between diagnoses."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.llm.llm_client import StructuredOutputError
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class DoctorDiffQuestionResponse(BaseModel):
    """Structured response for differential diagnostic question generation."""
    question: str = Field(..., description="Your specific diagnostic question")


class DoctorQDiffAgentOutput(AgentOutput):
    """Output from the doctor question differential agent."""
    question: str = Field(default="", description="ONE specific diagnostic question to distinguish between competing diagnoses")
    question_reasoning: str | None = Field(default=None, description="Brief rationale for the question (simple prompts only)")


class DoctorQDiffAgent(BaseAgent):
    """Agent that generates targeted questions to differentiate between competing diagnoses.
    
    The doctor question differential agent:
    - Takes the top 3 differential diagnoses with confidence scores
    - Analyzes the conversation history and structured summary
    - Asks ONE specific question that would most reduce uncertainty between diagnoses
    - Focuses only on distinguishing between competing diagnoses, not general screening
    - Prefers questions whose answers would strongly support one diagnosis AND weaken alternatives
    """
    
    agent_name = "doctor.q_diff"

    @property
    def _prompt_key(self) -> str:
        """Return the prompt key based on config flags."""
        has_turns = self.config.get("pass_turn_info", True)
        if not has_turns:
            return "doctor.q_diff_no_turns"
        return "doctor.q_diff"

    @property
    def system_prompt(self) -> str:
        """Return system prompt from the variant-aware prompt key."""
        return self._prompt_loader.get_system_prompt(self._prompt_key)

    async def execute(
        self,
        context: "WorkflowContext",
        predictions: list[dict] | None = None,
        str_summary: dict | str | None = None,
        current_round: int | None = None,
        max_rounds: int | None = None,
        feedback: str | None = None,
        rejected_question: str | None = None,
        **kwargs
    ) -> DoctorQDiffAgentOutput:
        """Generate the next differential diagnostic question.
        
        Args:
            context: Workflow context containing dialog history
            predictions: List of top 3 diagnoses with confidence scores
                        Each dict has 'disease' and 'confidence' keys
            str_summary: Structured summary from summarizer_str_agent
            current_round: Current round number (1-indexed)
            max_rounds: Maximum number of rounds allowed
            feedback: Optional feedback from repetition checker indicating
                     why the previous question was rejected
            
        Returns:
            DoctorQDiffAgentOutput with question, expected answers, and rationale
        """
        # Round 1: Ask for demographics if flag is enabled (default: True)
        if current_round == 1 and self.config.get("ask_demographics_first", True):
            return DoctorQDiffAgentOutput(
                success=True,
                question="What is your age, gender, and occupation?"
            )
        
        # Get explicit list of previously asked questions for deduplication
        include_asked_questions = self.config.get("pass_asked_questions", True)
        asked_questions = context.get_doctor_questions()
        if include_asked_questions and asked_questions:
            asked_questions_str = "\n".join(f"{i}. {q}" for i, q in enumerate(asked_questions, 1))
        else:
            asked_questions_str = ""
        
        # Get predictions - format as string for prompt
        if predictions:
            predictions_str = self._format_predictions(predictions)
        else:
            # Try to get from context
            predictions_str = ""
        
        # Get structured summary
        if str_summary is None:
            str_summary = context.get("summary", "")
        
        if isinstance(str_summary, dict):
            import json
            str_summary_str = json.dumps(str_summary, indent=2)
        else:
            str_summary_str = str(str_summary) if str_summary else ""
        
        # Format round information
        if current_round is not None and max_rounds is not None:
            rounds_remaining = max_rounds - current_round
            round_info = f"Round {current_round} of {max_rounds} ({rounds_remaining} questions remaining after this one)"
        else:
            round_info = "(Round information not available)"
        
        # Check if we have predictions
        if not predictions_str:
            return DoctorQDiffAgentOutput(
                success=False,
                error="No differential diagnoses provided",
                question=""
            )
        
        # Choose template: use _no_qlist variants when asked_questions is disabled
        template_suffix = "" if include_asked_questions else "_no_qlist"
        
        # Choose prompt template based on whether we have feedback
        if feedback:
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template_with_feedback",
                predictions=predictions_str,
                str_summary=str_summary_str,
                round_info=round_info,
                rejected_question=rejected_question or "(unknown)",
                feedback=feedback
            )
        else:
            template_name = f"user_prompt_template{template_suffix}"
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                template_name,
                predictions=predictions_str,
                str_summary=str_summary_str,
                asked_questions=asked_questions_str,
                round_info=round_info
            )

        try:
            response_model = DoctorDiffQuestionResponse
            result = await self._call_llm_structured(user_prompt, response_model)
            question = result.question
            
            # Clean up question
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            if question.startswith("'") and question.endswith("'"):
                question = question[1:-1]
            
            question = question.strip()
            if not question:
                return DoctorQDiffAgentOutput(
                    success=False,
                    error="LLM returned empty question",
                    question=""
                )
            
            q_reasoning = getattr(result, 'reasoning', None)
            return DoctorQDiffAgentOutput(
                success=True,
                question=question,
                question_reasoning=q_reasoning
            )
        except StructuredOutputError as e:
            # Fallback: use raw response directly as the question
            raw_question = e.raw_response.strip()
            # Clean up question
            if raw_question.startswith('"') and raw_question.endswith('"'):
                raw_question = raw_question[1:-1]
            if raw_question.startswith("'") and raw_question.endswith("'"):
                raw_question = raw_question[1:-1]
            if raw_question:
                print(f"  [FALLBACK] Using raw response as diff doctor question: {raw_question[:100]}...")
                return DoctorQDiffAgentOutput(
                    success=True,
                    question=raw_question
                )
            return DoctorQDiffAgentOutput(
                success=False,
                error=str(e),
                question=""
            )
        except Exception as e:
            print(f"  [LLM_ERROR] doctor_q_diff_agent failed: {str(e)}")
            return DoctorQDiffAgentOutput(
                success=False,
                error=str(e),
                question=""
            )
    
    def _format_predictions(self, predictions: list[dict]) -> str:
        """Format predictions list into a readable string, optionally with reasoning."""
        include_reasoning = self.config.get("pass_reasoning_to_diff", True)
        lines = []
        for i, pred in enumerate(predictions[:5], 1):
            disease = pred.get("disease", "Unknown")
            confidence = pred.get("confidence", 0)
            reasoning = pred.get("reasoning", "")
            lines.append(f"{i}. {disease} (confidence: {confidence}%)")
            if include_reasoning and reasoning:
                lines.append(f"   Reasoning: {reasoning}")
        return "\n".join(lines)
