"""Patient Agent - Answers doctor's questions based on patient history."""

from src.agents.base_agent import BaseAgent, AgentOutput
from src.llm.llm_client import StructuredOutputError
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class PatientResponse(BaseModel):
    """Structured response for patient answers."""
    answer: str = Field(..., description="The patient's answer to the doctor's question")


class PatientAgentOutput(AgentOutput):
    """Output from the patient agent."""
    answer: str = Field(default="", description="Patient's answer to the doctor's question")


class PatientAgent(BaseAgent):
    """Agent that simulates a patient answering doctor's questions.
    
    The patient agent:
    - Receives a detailed patient history and a doctor's question
    - Answers ONLY from information available in the patient history
    - Responds with "I don't know" if the answer is not in the history
    - Never asks questions, only responds
    - Answers succinctly and correctly
    """
    
    @property
    def agent_name(self) -> str:
        return "patient.default"

    async def execute(
        self,
        context: "WorkflowContext",
        patient_history: str | None = None,
        question: str | None = None,
        **kwargs
    ) -> PatientAgentOutput:
        """Answer the doctor's question based on patient history.
        
        Args:
            context: Workflow context (can contain patient_history)
            patient_history: The patient's medical history
            question: The doctor's question to answer
            
        Returns:
            PatientAgentOutput with the patient's answer
        """
        history = patient_history or context.get("patient_history", "")
        q = question or context.get("current_question", "")
        
        if not history:
            return PatientAgentOutput(success=False, error="No patient history provided", answer="")
        
        if not q:
            return PatientAgentOutput(success=False, error="No question provided", answer="")
        
        # Standard mode
        user_prompt = self.get_user_prompt(
            "user_prompt_template",
            patient_history=history,
            question=q
        )

        try:
            result = await self._call_llm_structured(user_prompt, PatientResponse)
            answer = result.answer.strip()
            
            if not answer:
                return PatientAgentOutput(
                    success=False,
                    error="LLM returned empty answer",
                    answer=""
                )
            
            return PatientAgentOutput(
                success=True,
                answer=answer
            )
        except StructuredOutputError as e:
            # Fallback: use raw response directly as the answer
            raw_answer = e.raw_response.strip()
            if raw_answer:
                print(f"  [FALLBACK] Using raw response as patient answer: {raw_answer[:100]}...")
                return PatientAgentOutput(
                    success=True,
                    answer=raw_answer
                )
            return PatientAgentOutput(
                success=False,
                error=str(e),
                answer=""
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc().replace('\n', ' | ')
            print(f"  [LLM_ERROR] patient_agent failed: {tb}")
            return PatientAgentOutput(
                success=False,
                error=tb,
                answer=""
            )
