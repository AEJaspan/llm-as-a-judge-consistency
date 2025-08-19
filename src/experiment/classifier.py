from pydantic import BaseModel
from typing import Dict, Type, Optional
from typing import Tuple
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.base_models import PredictionFloat, PredictionCategorical, PredictionInteger
from logger import logger

from config.constants import ModelProvider, APIConstants, ExperimentConstants


class LLMJudge:
    def __init__(
        self,
        model_name: str,
        temperature: float = APIConstants.DEFAULT_TEMPERATURE,
        max_retries: int = ExperimentConstants.DEFAULT_MAX_RETRIES,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.model = self._initialize_model()

    def _get_provider(self) -> ModelProvider:
        """Determine provider from model name using enum."""
        model_lower = self.model_name.lower()

        if "gpt" in model_lower or "openai" in model_lower:
            return ModelProvider.OPENAI
        elif "claude" in model_lower or "anthropic" in model_lower:
            return ModelProvider.ANTHROPIC
        elif "gemini" in model_lower or "google" in model_lower:
            return ModelProvider.GOOGLE
        else:
            raise ValueError(f"Unknown provider for model: {self.model_name}")

    def _initialize_model(self):
        """Initialize the appropriate LangChain model based on model_name"""
        provider = self._get_provider()

        if provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_retries=self.max_retries,
                timeout=ExperimentConstants.DEFAULT_TIMEOUT,  # Use constant
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _get_schema_and_description(
        self, confidence_type: str
    ) -> Tuple[Type[BaseModel], str]:
        """Get schema and description for confidence type."""
        schema_map = {
            "float": (PredictionFloat, "a float between 0.0 and 1.0"),
            "categorical": (
                PredictionCategorical,
                "one of: very low, low, medium, high, very high",
            ),
            "integer": (PredictionInteger, "an integer between 0 and 5"),
        }

        if confidence_type not in schema_map:
            raise ValueError(f"Unknown confidence_type: {confidence_type}")

        return schema_map[confidence_type]

    def _create_chain(self, confidence_type: str, system_prompt: Optional[str] = None):
        """Create the prompt template and chain for classification."""
        # Get schema and description
        schema, confidence_description = self._get_schema_and_description(
            confidence_type
        )

        # Create structured output model
        structured_model = self.model.with_structured_output(schema)

        # Create the prompt template
        system_message = (
            system_prompt
            or "Classify the sentiment as positive (true) or negative (false)."
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""{system_message}
            
You must analyze the given text and provide both a classification and your confidence in that classification.

Classification: Return true for positive sentiment, false for negative sentiment.
Confidence: Provide your confidence as {confidence_description}.

Be thoughtful about your confidence level - consider factors like:
- Clarity of sentiment indicators
- Presence of mixed signals
- Ambiguity in language
- Your certainty in the classification""",
                ),
                ("human", "Text to classify: {text}"),
            ]
        )

        return prompt_template | structured_model, schema

    def _get_fallback_response(self, schema: Type[BaseModel]) -> Dict:
        """Get fallback response for a given schema."""
        fallbacks = {
            PredictionFloat: {"classification": False, "confidence": 0.5},
            PredictionCategorical: {"classification": False, "confidence": "medium"},
            PredictionInteger: {"classification": False, "confidence": 2},
        }

        response = fallbacks[schema].copy()
        response["classification"] = None
        return response

    async def _execute_with_retry(
        self, chain, text: str, schema: Type[BaseModel]
    ) -> Dict:
        """Execute chain with retry logic."""
        for attempt in range(self.max_retries):
            try:
                result = await chain.ainvoke({"text": text})
                return {
                    "classification": result.classification,
                    "confidence": result.confidence,
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Final attempt failed for model {self.model_name}: {e}"
                    )
                    return self._get_fallback_response(schema)
                else:
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying... Error: {e}"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return self._get_fallback_response(schema)

    async def classify_with_confidence(
        self, text: str, confidence_type: str, system_prompt: Optional[str] = None
    ) -> Dict:
        """Async classification with confidence score."""
        try:
            chain, schema = self._create_chain(confidence_type, system_prompt)
            return await self._execute_with_retry(chain, text, schema)
        except ValueError as e:
            # Re-raise ValueError for invalid confidence_type
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in classify_with_confidence: {e}")
            # Get schema for fallback
            schema, _ = self._get_schema_and_description(confidence_type)
            return self._get_fallback_response(schema)

    def classify_with_confidence_sync(
        self, text: str, confidence_type: str, system_prompt: Optional[str] = None
    ) -> Dict:
        """Sync version of classify_with_confidence."""
        try:
            chain, schema = self._create_chain(confidence_type, system_prompt)

            try:
                result = chain.invoke({"text": text})
                return {
                    "classification": result.classification,
                    "confidence": result.confidence,
                }
            except Exception as e:
                logger.error(
                    f"Error with model {self.model_name}, confidence_type {confidence_type}: {e}"
                )
                return self._get_fallback_response(schema)

        except ValueError as e:
            # Re-raise ValueError for invalid confidence_type
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in classify_with_confidence_sync: {e}")
            # Get schema for fallback
            schema, _ = self._get_schema_and_description(confidence_type)
            return self._get_fallback_response(schema)
