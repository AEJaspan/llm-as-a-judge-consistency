import numpy as np
from pydantic import BaseModel
from typing import Dict
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.base_models import (
    PredictionFloat, 
    PredictionCategorical, 
    PredictionInteger
)


class LLMJudge:
    def __init__(self, model_name: str, api_key: str = None, temperature: float = 0.0, max_retries: int = 3):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the appropriate LangChain model based on model_name"""
        if "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_retries=self.max_retries
            )
        # elif "claude" in self.model_name.lower() or "anthropic" in self.model_name.lower():
        #     return ChatAnthropic(
        #         model=self.model_name,
        #         api_key=self.api_key,
        #         temperature=self.temperature,
        #         max_retries=self.max_retries
        #     )
        # elif "gemini" in self.model_name.lower() or "google" in self.model_name.lower():
        #     return ChatGoogleGenerativeAI(
        #         model=self.model_name,
        #         google_api_key=self.api_key,
        #         temperature=self.temperature
        #     )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    async def classify_with_confidence(self, text: str, confidence_type: str, system_prompt: str = None) -> Dict:
        """Classify text and return confidence score based on the specified format"""
        
        # Choose the appropriate schema and get structured output model
        if confidence_type == "float":
            schema = PredictionFloat
            confidence_description = "a float between 0.0 and 1.0"
        elif confidence_type == "categorical":
            schema = PredictionCategorical
            confidence_description = "one of: very low, low, medium, high, very high"
        elif confidence_type == "integer":
            schema = PredictionInteger
            confidence_description = "an integer between 0 and 5"
        else:
            raise ValueError(f"Unknown confidence_type: {confidence_type}")
        
        # Create structured output model
        structured_model = self.model.with_structured_output(schema)
        
        # Create the prompt template
        system_message = system_prompt or "Classify the sentiment as positive (true) or negative (false)."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""{system_message}
            
You must analyze the given text and provide both a classification and your confidence in that classification.

Classification: Return true for positive sentiment, false for negative sentiment.
Confidence: Provide your confidence as {confidence_description}.

Be thoughtful about your confidence level - consider factors like:
- Clarity of sentiment indicators
- Presence of mixed signals
- Ambiguity in language
- Your certainty in the classification"""),
            ("human", "Text to classify: {text}")
        ])
        
        # Create the chain and invoke with retry logic
        chain = prompt_template | structured_model
        
        for attempt in range(self.max_retries):
            try:
                result = await chain.ainvoke({"text": text})
                
                # Convert Pydantic model to dict
                return {
                    "classification": result.classification,
                    "confidence": result.confidence
                }
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Final attempt failed for model {self.model_name}, confidence_type {confidence_type}: {e}")
                    # Return a fallback response
                    return await self._fallback_response(schema)
                else:
                    print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Fallback (shouldn't reach here due to the loop logic above)
        return await self._fallback_response(schema)
    
    async def _fallback_response(self, schema: BaseModel) -> Dict:
        """Fallback response in case of API errors"""
        classification = np.random.choice([True, False])
        
        if schema == PredictionFloat:
            confidence = 0.5  # Neutral confidence for fallback
        elif schema == PredictionCategorical:
            confidence = "medium"
        else:  # integer
            confidence = 2  # Middle value for fallback
            
        return {"classification": classification, "confidence": confidence}
    
    def classify_with_confidence_sync(self, text: str, confidence_type: str, system_prompt: str = None) -> Dict:
        """Synchronous version for easier testing"""
        # Choose the appropriate schema
        if confidence_type == "float":
            schema = PredictionFloat
            confidence_description = "a float between 0.0 and 1.0"
        elif confidence_type == "categorical":
            schema = PredictionCategorical
            confidence_description = "one of: very low, low, medium, high, very high"
        elif confidence_type == "integer":
            schema = PredictionInteger
            confidence_description = "an integer between 0 and 5"
        else:
            raise ValueError(f"Unknown confidence_type: {confidence_type}")
        
        # Create structured output model
        structured_model = self.model.with_structured_output(schema)
        
        # Create the prompt template
        system_message = system_prompt or "Classify the sentiment as positive (true) or negative (false)."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""{system_message}
            
You must analyze the given text and provide both a classification and your confidence in that classification.

Classification: Return true for positive sentiment, false for negative sentiment.
Confidence: Provide your confidence as {confidence_description}.

Be thoughtful about your confidence level - consider factors like:
- Clarity of sentiment indicators
- Presence of mixed signals
- Ambiguity in language
- Your certainty in the classification"""),
            ("human", "Text to classify: {text}")
        ])
        
        # Create the chain and invoke
        chain = prompt_template | structured_model
        
        try:
            result = chain.invoke({"text": text})
            
            # Convert Pydantic model to dict
            return {
                "classification": result.classification,
                "confidence": result.confidence
            }
            
        except Exception as e:
            print(f"Error with model {self.model_name}, confidence_type {confidence_type}: {e}")
            # Return a fallback response
            return self._fallback_response_sync(schema)
    
    def _fallback_response_sync(self, schema: BaseModel) -> Dict:
        """Synchronous fallback response"""
        classification = np.random.choice([True, False])
        
        if schema == PredictionFloat:
            confidence = 0.5
        elif schema == PredictionCategorical:
            confidence = "medium"
        else:  # integer
            confidence = 2
            
        return {"classification": classification, "confidence": confidence}
