"""
LLM Provider abstraction using Pydantic AI
Supports multiple providers: OpenAI, Vertex AI, Gemini API (Gemma/Gemini), Anthropic

For models without function calling support (like Gemma 3), uses the Google GenAI SDK
directly with text-based JSON generation and Pydantic validation.
"""
import os
import json
import re
from typing import Optional, Type, TypeVar, Any, List
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.gemini import GeminiModel

from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)

# Models that don't support function calling/system instructions (use direct GenAI SDK)
MODELS_WITHOUT_FUNCTION_CALLING = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-2-27b-it",
    "gemma-2-9b-it",
    "gemma-2-2b-it",
]

# Set Google Application Credentials if configured
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.GOOGLE_APPLICATION_CREDENTIALS
    logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS from config")

T = TypeVar('T', bound=BaseModel)


class LLMProvider:
    """
    Unified LLM provider interface using Pydantic AI
    Supports OpenAI, Vertex AI (Gemma), and future providers
    """

    def __init__(self):
        self._model: Optional[Model] = None
        self._provider_name: str = settings.AI_PROVIDER

    @property
    def model(self) -> Model:
        """Get or create the LLM model based on configuration"""
        if self._model is None:
            self._model = self._create_model()
        return self._model

    @property
    def provider_name(self) -> str:
        """Get the current provider name"""
        return self._provider_name

    @property
    def supports_function_calling(self) -> bool:
        """Check if the current model supports function calling"""
        model_name = self._get_current_model_name()
        return model_name not in MODELS_WITHOUT_FUNCTION_CALLING

    def _get_current_model_name(self) -> str:
        """Get the current model name based on provider"""
        provider = settings.AI_PROVIDER.lower()
        if provider == "openai":
            return settings.OPENAI_MODEL
        elif provider == "vertex":
            return settings.VERTEX_MODEL
        elif provider == "gemini":
            return settings.GEMINI_MODEL
        elif provider == "anthropic":
            return settings.ANTHROPIC_MODEL
        return ""

    def _create_model(self) -> Model:
        """Create the appropriate model based on AI_PROVIDER setting"""
        provider = settings.AI_PROVIDER.lower()

        if provider == "openai":
            return self._create_openai_model()
        elif provider == "vertex":
            return self._create_vertex_model()
        elif provider == "gemini":
            return self._create_gemini_model()
        elif provider == "anthropic":
            return self._create_anthropic_model()
        else:
            raise ValueError(f"Unsupported AI provider: {provider}. Supported: openai, vertex, gemini, anthropic")

    def _create_openai_model(self) -> OpenAIChatModel:
        """Create OpenAI model"""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")

        # Set API key in environment for pydantic-ai to pick up
        os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

        model_name = settings.OPENAI_MODEL
        logger.info(f"Initializing OpenAI model: {model_name}")

        return OpenAIChatModel(model_name)

    def _create_vertex_model(self) -> GeminiModel:
        """Create Vertex AI model (Gemini/Gemma via Vertex AI)"""
        if not settings.VERTEX_PROJECT:
            raise ValueError("VERTEX_PROJECT not configured for Vertex AI")

        model_name = settings.VERTEX_MODEL

        logger.info(f"Initializing Vertex AI model: {model_name} (project={settings.VERTEX_PROJECT}, location={settings.VERTEX_LOCATION})")

        # Use GeminiModel with project_id for Vertex AI
        return GeminiModel(
            model_name,
            project_id=settings.VERTEX_PROJECT,
            region=settings.VERTEX_LOCATION
        )

    def _create_gemini_model(self) -> GeminiModel:
        """Create Gemini API model (Gemini/Gemma via Google AI Studio API key)"""
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not configured for Gemini API")

        model_name = settings.GEMINI_MODEL

        logger.info(f"Initializing Gemini API model: {model_name}")

        return GeminiModel(model_name, api_key=settings.GOOGLE_API_KEY)

    def _create_anthropic_model(self) -> Model:
        """Create Anthropic model"""
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        # Import Anthropic model only when needed
        from pydantic_ai.models.anthropic import AnthropicModel

        model_name = settings.ANTHROPIC_MODEL
        logger.info(f"Initializing Anthropic model: {model_name}")

        return AnthropicModel(
            model_name,
            api_key=settings.ANTHROPIC_API_KEY
        )

    def create_agent(
        self,
        output_type: Type[T],
        system_prompt: str,
        name: str = "Agent"
    ) -> Agent[None, T]:
        """
        Create a Pydantic AI agent with the configured model

        Args:
            output_type: Pydantic model class for structured output
            system_prompt: System instructions for the agent
            name: Agent name for logging

        Returns:
            Configured Pydantic AI Agent
        """
        logger.info(f"Creating agent '{name}' with provider: {self._provider_name}")

        return Agent(
            self.model,
            output_type=output_type,
            system_prompt=system_prompt
        )

    async def generate_structured(
        self,
        output_type: Type[T],
        system_prompt: str,
        user_prompt: str
    ) -> T:
        """
        Generate structured output using the configured LLM

        Args:
            output_type: Pydantic model class for structured output
            system_prompt: System instructions
            user_prompt: User message/prompt

        Returns:
            Structured output as Pydantic model instance
        """
        agent = self.create_agent(output_type, system_prompt)
        result = await agent.run(user_prompt)
        return result.output

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Generate text output using the configured LLM

        Args:
            system_prompt: System instructions
            user_prompt: User message/prompt

        Returns:
            Generated text response
        """
        agent = Agent(self.model, output_type=str, system_prompt=system_prompt)
        result = await agent.run(user_prompt)
        return result.output

    async def generate_structured_with_fallback(
        self,
        output_type: Type[T],
        system_prompt: str,
        user_prompt: str
    ) -> T:
        """
        Generate structured output with fallback for models without function calling.

        For models that support function calling (OpenAI, Gemini, Claude), uses
        Pydantic AI's native structured output.

        For models without function calling (Gemma), uses text-based JSON generation
        with manual parsing and Pydantic validation.

        Args:
            output_type: Pydantic model class for structured output
            system_prompt: System instructions
            user_prompt: User message/prompt

        Returns:
            Structured output as Pydantic model instance
        """
        if self.supports_function_calling:
            # Use native structured output
            logger.info(f"Using native function calling for {self._get_current_model_name()}")
            return await self.generate_structured(output_type, system_prompt, user_prompt)
        else:
            # Use text-based JSON fallback
            logger.info(f"Using JSON text fallback for {self._get_current_model_name()} (no function calling)")
            return await self._generate_structured_via_json(output_type, system_prompt, user_prompt)

    async def _generate_structured_via_json(
        self,
        output_type: Type[T],
        system_prompt: str,
        user_prompt: str
    ) -> T:
        """
        Generate structured output by asking the model to return JSON and parsing it.
        Uses the Google GenAI SDK directly for Gemma models (which don't support
        function calling or system instructions via Pydantic AI).
        """
        # Get the JSON schema from the Pydantic model
        schema = output_type.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Combine system prompt and user prompt into a single prompt
        # (Gemma doesn't support system instructions, so we include it in the user message)
        combined_prompt = f"""{system_prompt}

---

{user_prompt}

---

CRITICAL INSTRUCTIONS:
1. You MUST respond with ONLY valid JSON that matches this exact schema:
{schema_str}

2. Do NOT include any text before or after the JSON.
3. Do NOT use markdown code blocks (no ```json or ```).
4. Just output the raw JSON object starting with {{ and ending with }}."""

        # Use Google GenAI SDK directly for Gemma models
        text_response = await self._call_gemma_direct(combined_prompt)

        # Parse and validate the JSON response
        return self._parse_json_response(text_response, output_type)

    async def _call_gemma_direct(self, prompt: str) -> str:
        """
        Call Gemma model directly using Google GenAI SDK.
        This bypasses Pydantic AI which doesn't work well with Gemma's limitations.
        """
        try:
            from google import genai

            logger.info(f"Calling Gemma directly via GenAI SDK: {self._get_current_model_name()}")

            # Create client with API key
            client = genai.Client(api_key=settings.GOOGLE_API_KEY)

            # Generate content
            response = client.models.generate_content(
                model=self._get_current_model_name(),
                contents=prompt,
            )

            # Extract text from response
            result_text = response.text
            logger.debug(f"Gemma response length: {len(result_text)} chars")

            return result_text

        except ImportError:
            raise ValueError("google-genai package not installed. Run: pip install google-genai")
        except Exception as e:
            logger.error(f"Gemma direct call failed: {e}")
            raise

    def _parse_json_response(self, text: str, output_type: Type[T]) -> T:
        """
        Parse JSON from text response and validate with Pydantic.
        Handles common formatting issues from LLMs.
        """
        # Clean up the response
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try to extract JSON object if there's surrounding text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group()

        try:
            # Parse JSON
            data = json.loads(text)

            # Validate with Pydantic
            return output_type.model_validate(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {text[:500]}...")
            raise ValueError(f"Model returned invalid JSON: {e}")

        except ValidationError as e:
            logger.error(f"JSON validation failed: {e}")
            logger.error(f"Parsed data: {data}")
            raise ValueError(f"Model returned JSON that doesn't match schema: {e}")


# Singleton instance
_llm_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get or create singleton LLM provider instance"""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider()
    return _llm_provider


def reset_llm_provider():
    """Reset the singleton (useful for testing or config changes)"""
    global _llm_provider
    _llm_provider = None
