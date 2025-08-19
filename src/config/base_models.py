
from pydantic import BaseModel, Field

from typing import Literal, List
from enum import Enum
from dataclasses import dataclass
from config.constants import ExperimentConstants


class PredictionFloat(BaseModel):
    classification: bool = Field(..., description="the binary classification of the text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="the confidence in the classification")

class PredictionCategorical(BaseModel):
    classification: bool = Field(..., description="the binary classification of the text")
    confidence: Literal["very low", "low", "medium", "high", "very high"] = Field(..., description="the confidence in the classification")

class PredictionInteger(BaseModel):
    classification: bool = Field(..., description="the binary classification of the text")
    confidence: int = Field(..., ge=0, le=5, description="the confidence in the classification")

# Dataset configuration enum
class DatasetChoice(Enum):
    SST2 = "sst2"
    SMS_SPAM = "sms_spam"

# Dataset configurations
DATASET_CONFIGS = {
    DatasetChoice.SST2: {
        "hf_name": "stanfordnlp/sst2",
        "hf_config": None,
        "text_column": "sentence",
        "label_column": "label",
        "description": "Stanford Sentiment Treebank - Binary sentiment classification",
        "label_meanings": {0: "negative", 1: "positive"},
        "task_description": "Classify the sentiment as positive (true) or negative (false)."
    },
    DatasetChoice.SMS_SPAM: {
        "hf_name": "ucirvine/sms_spam", 
        "hf_config": None,
        "text_column": "sms",
        "label_column": "label", 
        "description": "SMS Spam Detection - Ham vs Spam classification",
        "label_meanings": {0: "ham (legitimate)", 1: "spam"},
        "task_description": "Classify the SMS message as spam (true) or legitimate/ham (false)."
    }
}

# Mapping for categorical confidence to numerical values
CATEGORICAL_TO_FLOAT = {
    "very low": .0/5,
    "low": 1.0/5,
    "medium": 2.0/5,
    "high": 3.0/5,
    "very high": 4.0/5
}
@dataclass
class ExperimentConfig:
    dataset_choice: DatasetChoice
    sample_size: int = ExperimentConstants.MIN_SAMPLE_SIZE  # Use constant instead of 1000
    models: List[str] = None
    confidence_types: List[str] = None
    def __post_init__(self):
        # Validate sample size using constants
        if self.sample_size < ExperimentConstants.MIN_SAMPLE_SIZE:
            raise ValueError(f"Sample size must be at least {ExperimentConstants.MIN_SAMPLE_SIZE}")
        if self.sample_size > ExperimentConstants.MAX_SAMPLE_SIZE:
            raise ValueError(f"Sample size cannot exceed {ExperimentConstants.MAX_SAMPLE_SIZE}")

    @property
    def dataset_config(self):
        """Get the configuration for the selected dataset"""
        return DATASET_CONFIGS[self.dataset_choice]
    
    @property
    def dataset_name(self):
        return self.dataset_config["hf_name"]
    
    @property
    def dataset_config_name(self):
        return self.dataset_config["hf_config"]
    
    @property
    def text_column(self):
        return self.dataset_config["text_column"]
    
    @property
    def label_column(self):
        return self.dataset_config["label_column"]
    
    @property
    def task_description(self):
        return self.dataset_config["task_description"]
    
    @property
    def experiment_name(self):
        """Generate a descriptive name for this experiment configuration"""
        return f"{self.dataset_choice.value}_confidence_experiment"
