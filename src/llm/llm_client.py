"""LLM client abstraction supporting multiple providers."""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from src.config import get_settings

T = TypeVar('T', bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when structured output parsing fails after all retries.
    
    Includes the last raw response so callers can decide how to handle it.
    """
    def __init__(self, message: str, raw_response: str, last_error: Exception | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.last_error = last_error


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, temperature: float = 0, max_tokens: int = 4096, unstructured_mode: bool = False):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.unstructured_mode = unstructured_mode
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            system_prompt: The system/instruction prompt.
            user_prompt: The user input prompt.
            
        Returns:
            The generated response as a string.
        """
        pass
    
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T]
    ) -> T:
        """Generate a structured response from the LLM using Pydantic model.
        
        Args:
            system_prompt: The system/instruction prompt.
            user_prompt: The user input prompt.
            response_model: Pydantic model class to parse the response into.
            
        Returns:
            Parsed Pydantic model instance.
        """
        # Use unstructured mode with retry if enabled
        if self.unstructured_mode:
            print("USING UNSTRUCTURED MODE WITH RETRY")
            return await self._generate_structured_with_retry(
                system_prompt, user_prompt, response_model
            )
        print("USING NATIVE STRUCTURED OUTPUT PARSING")
        # Default implementation: use generate_json and parse
        json_response = await self.generate_json(system_prompt, user_prompt)
        return response_model.model_validate(json_response)
    
    async def _generate_structured_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_retries: int = 3
    ) -> T:
        """Generate structured output in unstructured mode with retry on parse failure.
        
        Calls generate() normally (no response_format), then validates JSON against
        the Pydantic model. On failure, retries with a reminder to follow the structure.
        
        Args:
            system_prompt: The system/instruction prompt.
            user_prompt: The user input prompt.
            response_model: Pydantic model class to parse the response into.
            max_retries: Maximum number of retry attempts (default 3).
            
        Returns:
            Parsed Pydantic model instance.
            
        Raises:
            StructuredOutputError: If all retries fail, includes the last raw response.
        """
        current_user_prompt = user_prompt
        last_error = None
        last_raw_response = ""
        
        for attempt in range(max_retries):
            try:
                # Call generate normally (unstructured)
                response = await self.generate(system_prompt, current_user_prompt)
                last_raw_response = response  # Save in case we need it later
                
                # Try to parse JSON from response
                # print("Raw response for structured output parsing:", response)
                # print("\nDone\n")
                json_response = self._parse_json_response(response)
                # print("hi there", json_response)
                # print("\ndone\n")
                # Validate against Pydantic model
                return response_model.model_validate(json_response)
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Append reminder to follow the structure for next attempt
                    current_user_prompt = user_prompt + "\n\nFOLLOW THE JSON STRUCTURE MENTIONED ABOVE STRICTLY."
                    print(f"  [RETRY {attempt + 1}/{max_retries}] JSON parse/validation failed: {str(e)[:100]}")
                else:
                    print(f"  [FAILED] All {max_retries} attempts failed for structured output")
        
        # All retries exhausted - raise custom exception with raw response
        raise StructuredOutputError(
            f"Failed to generate valid structured output after {max_retries} attempts: {last_error}",
            raw_response=last_raw_response,
            last_error=last_error
        )
    
    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate a JSON response from the LLM.
        
        Args:
            system_prompt: The system/instruction prompt.
            user_prompt: The user input prompt.
            
        Returns:
            The parsed JSON response as a dictionary.
        """
        # Append JSON instruction to system prompt
        json_system_prompt = system_prompt + "\n\nIMPORTANT: You must respond with valid JSON only. No markdown code blocks, no explanations outside the JSON."
        
        response = await self.generate(json_system_prompt, user_prompt)
        
        # Try to extract JSON from the response
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        # Try direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If all else fails, raise an error
        raise ValueError(f"Could not parse JSON from response: {response[:200]}...")


class OpenAIClient(LLMClient):
    """OpenAI-compatible client.

    Subclassed by :class:`GrokClient`, :class:`GeminiClient`, :class:`DeepSeekClient`
    which only change ``_base_url`` (and ``_token_limit_kwarg`` where needed).
    """

    # Override in subclasses. ``None`` = OpenAI default endpoint.
    _base_url: str | None = None
    # OpenAI native uses ``max_completion_tokens``; xAI / Gemini-OpenAI-compat /
    # DeepSeek currently expect ``max_tokens``. Override per-subclass.
    _token_limit_kwarg: str = "max_completion_tokens"

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        from openai import AsyncOpenAI
        # Precedence: explicit constructor arg > subclass class attribute > SDK default.
        effective_base_url = base_url or self._base_url
        if effective_base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=effective_base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        self.deployment_name = model

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def _is_o_series(self) -> bool:
        """OpenAI o-series reasoning models (o1, o3, o4)."""
        name = self.deployment_name.lower()
        return any(f"o{n}" in name for n in ("1", "3", "4"))

    @property
    def _token_limit_kwargs(self) -> dict:
        # o-series reasoning tokens count against the limit, so give them more headroom
        limit = 8192 if self._is_o_series else self.max_tokens
        return {self._token_limit_kwarg: limit}

    def _temp_kwargs(self, effective_temp: float) -> dict:
        """o-series only accepts the default temperature (=1); omit ours."""
        if self._is_o_series:
            return {}
        return {"temperature": effective_temp}

    def get_turn_tokens(self) -> tuple[int, int]:
        return self.turn_input_tokens, self.turn_output_tokens

    def reset_turn_tokens(self):
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0

    def get_total_tokens(self) -> tuple[int, int]:
        return self.total_input_tokens, self.total_output_tokens

    def _track_usage(self, usage):
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self.turn_input_tokens += usage.prompt_tokens
            self.turn_output_tokens += usage.completion_tokens

    async def _retry_with_backoff(self, func, max_retries: int = 3):
        """Execute ``func`` with 30s / 60s / 90s retry on any error."""
        delays = [30, 60, 90]
        last_exception = None
        for attempt in range(max_retries):
            try:
                result = await func()
                if attempt > 0:
                    print(f"  [RECOVERED] {self.provider} call succeeded on attempt {attempt + 1}/{max_retries}")
                return result
            except Exception as e:
                last_exception = e
                delay = delays[attempt]
                print(f"  [RETRY] {str(e)[:100]}... retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
        raise last_exception

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        effective_temp = temperature if temperature is not None else self.temperature

        async def _call():
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **self._temp_kwargs(effective_temp),
                **self._token_limit_kwargs,
            )
            self._track_usage(response.usage)
            return response.choices[0].message.content or ""

        return await self._retry_with_backoff(_call)

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float | None = None,
    ) -> T:
        """Native structured output via ``beta.chat.completions.parse``.

        After the native call has exhausted its 3 retries, fall back to a single
        attempt of ``_generate_structured_with_retry`` (plain-text generation +
        JSON parse + Pydantic validation).
        """
        effective_temp = temperature if temperature is not None else self.temperature

        if self.unstructured_mode:
            return await self._generate_structured_with_retry(
                system_prompt, user_prompt, response_model
            )

        async def _call():
            response = await self.client.beta.chat.completions.parse(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                **self._temp_kwargs(effective_temp),
                **self._token_limit_kwargs,
            )
            self._track_usage(response.usage)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError(f"Failed to parse structured response: {response.choices[0].message.content}")
            return parsed

        try:
            return await self._retry_with_backoff(_call)
        except Exception as e:
            print(
                f"  [{self.provider.upper()}] Native structured output failed after retries "
                f"({str(e)[:100]}); falling back to unstructured retry (1 attempt)"
            )
            return await self._generate_structured_with_retry(
                system_prompt, user_prompt, response_model, max_retries=1
            )


class GrokClient(OpenAIClient):
    """xAI Grok client via the OpenAI-compatible endpoint (https://api.x.ai/v1)."""
    _base_url = "https://api.x.ai/v1"
    _token_limit_kwarg = "max_tokens"

    def __init__(self, api_key: str, model: str = "grok-4-1", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)

    @property
    def provider(self) -> str:
        return "grok"


class GeminiClient(OpenAIClient):
    """Google Gemini client via the OpenAI-compatible endpoint
    (https://generativelanguage.googleapis.com/v1beta/openai/)."""
    _base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    _token_limit_kwarg = "max_tokens"

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)

    @property
    def provider(self) -> str:
        return "gemini"


class DeepSeekClient(OpenAIClient):
    """DeepSeek client via the OpenAI-compatible endpoint (https://api.deepseek.com/v1).

    DeepSeek does not accept OpenAI's ``response_format={"type": "json_schema", ...}``,
    so :meth:`generate_structured` injects the Pydantic model's JSON schema into
    the system prompt and requests ``response_format={"type": "json_object"}``.
    If that call fails (network error, malformed JSON, schema violation, ...) we
    fall back to :meth:`_generate_structured_with_retry`.
    """
    _base_url = "https://api.deepseek.com/v1"
    _token_limit_kwarg = "max_tokens"

    def __init__(self, api_key: str, model: str = "Deepseek-V3.2", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)

    @property
    def provider(self) -> str:
        return "deepseek"

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float | None = None,
    ) -> T:
        """DeepSeek JSON mode with schema in the prompt; fall back to retry-and-validate on failure."""
        effective_temp = temperature if temperature is not None else self.temperature

        async def _call():
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                **self._temp_kwargs(effective_temp),
                **self._token_limit_kwargs,
            )
            self._track_usage(response.usage)
            content = response.choices[0].message.content or ""
            return response_model.model_validate_json(content)

        try:
            return await _call()
        except Exception as e:
            print(
                f"  [DEEPSEEK] JSON-mode call failed ({str(e)[:100]}); "
                f"falling back to unstructured retry"
            )
            return await self._generate_structured_with_retry(
                system_prompt, user_prompt, response_model
            )


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs
) -> LLMClient:
    """Factory function to create an LLM client based on provider.

    Args:
        provider: One of 'openai', 'grok', 'gemini', 'deepseek'.
                  Defaults to settings.
        model: Model name. Defaults to settings.
        api_key: API key. Defaults to settings based on provider.
        **kwargs: Additional arguments passed to the client.

    Returns:
        An LLM client instance.
    """
    settings = get_settings()

    provider = provider or settings.llm_provider
    model = model or settings.llm_model

    # Set default kwargs from settings
    kwargs.setdefault("temperature", settings.temperature)
    kwargs.setdefault("max_tokens", settings.max_tokens)
    kwargs.setdefault("unstructured_mode", settings.unstructured_mode)

    if provider == "openai":
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        kwargs.setdefault("base_url", settings.openai_base_url)
        return OpenAIClient(api_key=api_key, model=model, **kwargs)

    elif provider == "gemini":
        api_key = api_key or settings.google_api_key
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        kwargs.setdefault("base_url", settings.gemini_base_url)
        return GeminiClient(api_key=api_key, model=model, **kwargs)

    elif provider == "grok":
        api_key = api_key or settings.xai_api_key
        if not api_key:
            raise ValueError("xAI API key is required. Set XAI_API_KEY environment variable.")
        kwargs.setdefault("base_url", settings.xai_base_url)
        return GrokClient(api_key=api_key, model=model, **kwargs)

    elif provider == "deepseek":
        api_key = api_key or settings.deepseek_api_key
        if not api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable.")
        kwargs.setdefault("base_url", settings.deepseek_base_url)
        return DeepSeekClient(api_key=api_key, model=model, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Supported: openai, gemini, grok, deepseek"
        )
