"""Doctor Question Agent - Generates diagnostic questions for the patient."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.llm.llm_client import StructuredOutputError
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class DoctorQuestionResponse(BaseModel):
    """Structured response for doctor question generation."""
    question: str = Field(..., description="The doctor's diagnostic question for the patient")


class DoctorQAgentOutput(AgentOutput):
    """Output from the doctor question agent."""
    question: str = Field(default="", description="The doctor's question for the patient")
    question_reasoning: str | None = Field(default=None, description="Brief rationale for the question (simple prompts only)")


class DoctorQAgent(BaseAgent):
    """Agent that simulates a doctor asking diagnostic questions.
    
    The doctor question agent:
    - Analyzes the conversation history so far
    - Asks relevant questions to narrow down the diagnosis
    - Asks simple, atomic questions not previously asked
    - Can ask about demographics, symptoms, history, physical exam, tests
    """
    
    agent_name = "doctor.q"

    @property
    def _prompt_key(self) -> str:
        """Return the prompt key based on config flags."""
        has_turns = self.config.get("pass_turn_info", True)
        if not has_turns:
            return "doctor.q_no_turns"
        return "doctor.q"

    @property
    def system_prompt(self) -> str:
        """Return system prompt from the variant-aware prompt key."""
        return self._prompt_loader.get_system_prompt(self._prompt_key)

    async def execute(
        self,
        context: "WorkflowContext",
        dialog: str | None = None,
        current_round: int | None = None,
        max_rounds: int | None = None,
        feedback: str | None = None,
        rejected_question: str | None = None,
        temperature: float | None = None,
        **kwargs
    ) -> DoctorQAgentOutput:
        """Generate the next diagnostic question.
        
        Args:
            context: Workflow context containing dialog history
            dialog: Optional dialog string override
            current_round: Current round number (1-indexed)
            max_rounds: Maximum number of rounds allowed
            feedback: Optional feedback from repetition checker indicating
                     why the previous question was rejected
            temperature: Optional temperature override for this call
            
        Returns:
            DoctorQAgentOutput with the doctor's question
        """
        # Round 1: Ask for demographics if flag is enabled (default: True)
        if current_round == 1 and self.config.get("ask_demographics_first", True):
            return DoctorQAgentOutput(
                success=True,
                question="What is your age, gender, and occupation?"
            )
        
        # Get dialog from context or parameter
        if dialog:
            dialog_str = dialog
        elif context.dialog and context.dialog.round_count() > 0:
            dialog_str = context.dialog.format_readable()
        else:
            dialog_str = ""
        
        # Get explicit list of previously asked questions for deduplication
        include_asked_questions = self.config.get("pass_asked_questions", True)
        asked_questions = context.get_doctor_questions()
        if include_asked_questions and asked_questions:
            asked_questions_str = "\n".join(f"{i}. {q}" for i, q in enumerate(asked_questions, 1))
        else:
            asked_questions_str = ""
        
        # Format round information
        if current_round is not None and max_rounds is not None:
            rounds_remaining = max_rounds - current_round
            round_info = f"Round {current_round} of {max_rounds} ({rounds_remaining} questions remaining after this one)"
        else:
            round_info = "(Round information not available)"
        
        # Choose template: use _no_qlist variants when asked_questions is disabled
        template_suffix = "" if include_asked_questions else "_no_qlist"
        
        if not dialog_str or dialog_str.strip() == "":
            # First question - no prior conversation
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template_first",
                round_info=round_info
            )
        elif feedback:
            # Previous question was rejected - use feedback template
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                "user_prompt_template_with_feedback",
                dialog=dialog_str,
                round_info=round_info,
                rejected_question=rejected_question or "(unknown)",
                feedback=feedback
            )
        else:
            template_name = f"user_prompt_template{template_suffix}"
            user_prompt = self._prompt_loader.format_user_prompt(
                self._prompt_key,
                template_name,
                dialog=dialog_str,
                asked_questions=asked_questions_str,
                round_info=round_info
            )

        try:
            response_model = DoctorQuestionResponse
            result = await self._call_llm_structured(user_prompt, response_model, temperature=temperature)
            question = result.question.strip()
            
            # Remove any quotation marks if the model wrapped the question
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            if question.startswith("'") and question.endswith("'"):
                question = question[1:-1]
            
            if not question:
                return DoctorQAgentOutput(
                    success=False,
                    error="LLM returned empty question",
                    question=""
                )
            
            q_reasoning = getattr(result, 'reasoning', None)
            return DoctorQAgentOutput(
                success=True,
                question=question,
                question_reasoning=q_reasoning
            )
        except StructuredOutputError as e:
            # Fallback: use raw response directly as the question
            raw_question = e.raw_response.strip()
            # Remove any quotation marks if the model wrapped the question
            if raw_question.startswith('"') and raw_question.endswith('"'):
                raw_question = raw_question[1:-1]
            if raw_question.startswith("'") and raw_question.endswith("'"):
                raw_question = raw_question[1:-1]
            if raw_question:
                print(f"  [FALLBACK] Using raw response as doctor question: {raw_question[:100]}...")
                return DoctorQAgentOutput(
                    success=True,
                    question=raw_question
                )
            return DoctorQAgentOutput(
                success=False,
                error=str(e),
                question=""
            )
        except Exception as e:
            print(f"  [LLM_ERROR] doctor_q_agent failed: {str(e)}")
            return DoctorQAgentOutput(
                success=False,
                error=str(e),
                question=""
            )
