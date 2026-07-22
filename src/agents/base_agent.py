"""Base agent abstraction for all LLM-based agents."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

from src.prompts import get_prompt_loader

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.workflows.workflow_context import WorkflowContext

T = TypeVar('T', bound=BaseModel)


class AgentInput(BaseModel):
    """Base class for agent inputs - extend per agent."""
    pass


class AgentOutput(BaseModel):
    """Base class for agent outputs - extend per agent."""
    success: bool = True
    error: str | None = None


class BaseAgent(ABC):
    """Abstract base class for all agents in the medical diagnosis system.
    
    Each agent must implement:
    - execute(): The main logic for the agent
    - agent_name: The name used to look up prompts from prompts.json
    """
    
    # Override this in subclasses to specify the agent name for prompt lookup
    agent_name: str = "base_agent"
    
    def __init__(self, llm_client: "LLMClient", config: dict | None = None):
        """Initialize the agent with an LLM client and optional config.
        
        Args:
            llm_client: The LLM client to use for generating responses.
            config: Optional configuration dictionary for the agent.
        """
        self.llm = llm_client
        self.config = config or {}
        self.name = self.__class__.__name__
        self._prompt_loader = get_prompt_loader()
        # Per-agent token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent from the prompts file."""
        return self._prompt_loader.get_system_prompt(self.agent_name)
    
    def get_user_prompt(self, template_name: str = "user_prompt_template", **kwargs) -> str:
        """Get a formatted user prompt from the prompts file.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to format into the template
            
        Returns:
            The formatted user prompt
        """
        return self._prompt_loader.format_user_prompt(self.agent_name, template_name, **kwargs)
    
    @abstractmethod
    async def execute(self, context: "WorkflowContext", **kwargs) -> AgentOutput:
        """Execute the agent's main logic.
        
        Args:
            context: The workflow context containing shared state.
            **kwargs: Additional arguments specific to the agent.
            
        Returns:
            AgentOutput with the result of the agent's execution.
        """
        pass
    
    def _snapshot_llm_tokens(self) -> tuple[int, int]:
        """Snapshot the LLM client's current total tokens."""
        if hasattr(self.llm, 'get_total_tokens'):
            return self.llm.get_total_tokens()
        return 0, 0

    def _record_token_delta(self, before: tuple[int, int]):
        """Record the token delta since the snapshot."""
        after = self._snapshot_llm_tokens()
        d_in = after[0] - before[0]
        d_out = after[1] - before[1]
        self.total_input_tokens += d_in
        self.total_output_tokens += d_out
        self.turn_input_tokens += d_in
        self.turn_output_tokens += d_out

    def reset_turn_tokens(self):
        """Reset per-turn counters (called at the start of each round)."""
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0

    def reset_total_tokens(self):
        """Reset total counters (called at the start of each case)."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def get_turn_tokens(self) -> tuple[int, int]:
        return self.turn_input_tokens, self.turn_output_tokens

    def get_total_tokens(self) -> tuple[int, int]:
        return self.total_input_tokens, self.total_output_tokens

    # async def _call_llm(self, user_prompt: str, system_prompt: str | None = None, temperature: float | None = None) -> str:
    #     """Helper method to call the LLM with the agent's system prompt.
        
    #     Args:
    #         user_prompt: The user/input prompt to send to the LLM.
    #         system_prompt: Optional override for the system prompt.
    #         temperature: Optional temperature override for this call.
            
    #     Returns:
    #         The LLM's response as a string.
    #     """
    #     before = self._snapshot_llm_tokens()
    #     result = await self.llm.generate(
    #         system_prompt=system_prompt or self.system_prompt,
    #         user_prompt=user_prompt,
    #         temperature=temperature,
    #     )
    #     self._record_token_delta(before)
    #     return result
    
    # async def _call_llm_json(self, user_prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
    #     """Helper method to call the LLM and parse JSON response.
        
    #     Args:
    #         user_prompt: The user/input prompt to send to the LLM.
    #         system_prompt: Optional override for the system prompt.
            
    #     Returns:
    #         The parsed JSON response as a dictionary.
    #     """
    #     before = self._snapshot_llm_tokens()
    #     result = await self.llm.generate_json(
    #         system_prompt=system_prompt or self.system_prompt,
    #         user_prompt=user_prompt,
    #     )
    #     self._record_token_delta(before)
    #     return result
    
    async def _call_llm_structured(
        self,
        user_prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        temperature: float | None = None
    ) -> T:
        """Helper method to call the LLM with structured output parsing.
        
        Args:
            user_prompt: The user/input prompt to send to the LLM.
            response_model: Pydantic model class to parse the response into.
            system_prompt: Optional override for the system prompt.
            temperature: Optional temperature override for this call.
            
        Returns:
            Parsed Pydantic model instance.
        """
        before = self._snapshot_llm_tokens()
        result = await self.llm.generate_structured(
            system_prompt=system_prompt or self.system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            temperature=temperature,
        )
        self._record_token_delta(before)
        return result
    
    def __repr__(self) -> str:
        return f"{self.name}(llm={self.llm.provider})"
