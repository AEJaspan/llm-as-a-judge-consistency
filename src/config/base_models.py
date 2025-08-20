from pydantic import BaseModel, Field, validator
from typing import Literal, List, Optional
from enum import Enum
from config.constants import ExperimentConstants
import yaml
from pathlib import Path


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
    "very low": 0.0/5,
    "low": 1.0/5,
    "medium": 2.0/5,
    "high": 3.0/5,
    "very high": 4.0/5
}


class ExperimentConfig(BaseModel):
    """Pydantic model for experiment configuration with validation"""
    
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    dataset_choice: str = Field(..., description="Dataset to use for experiment")
    sample_size: int = Field(default=ExperimentConstants.MIN_SAMPLE_SIZE, description="Number of samples to use")
    models: List[str] = Field(default=["gpt-4o-mini"], description="List of models to test")
    confidence_types: List[str] = Field(default=["float"], description="List of confidence types to test")
    trials_per_config: int = Field(default=3, description="Number of trials per configuration")
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent API requests")
    
    @validator('dataset_choice')
    def validate_dataset_choice(cls, v):
        """Validate dataset choice against available options"""
        valid_choices = [choice.value for choice in DatasetChoice]
        if v not in valid_choices:
            raise ValueError(f"dataset_choice must be one of {valid_choices}, got {v}")
        return v
    
    @validator('sample_size')
    def validate_sample_size(cls, v):
        """Validate sample size within acceptable bounds"""
        if v < ExperimentConstants.MIN_SAMPLE_SIZE:
            raise ValueError(f"Sample size must be at least {ExperimentConstants.MIN_SAMPLE_SIZE}")
        if v > ExperimentConstants.MAX_SAMPLE_SIZE:
            raise ValueError(f"Sample size cannot exceed {ExperimentConstants.MAX_SAMPLE_SIZE}")
        return v
    
    @validator('confidence_types')
    def validate_confidence_types(cls, v):
        """Validate confidence types against supported formats"""
        valid_types = ["float", "categorical", "integer"]
        for conf_type in v:
            if conf_type not in valid_types:
                raise ValueError(f"confidence_type must be one of {valid_types}, got {conf_type}")
        return v
    
    @validator('models')
    def validate_models_not_empty(cls, v):
        """Ensure at least one model is specified"""
        if not v:
            raise ValueError("At least one model must be specified")
        return v
    
    @validator('trials_per_config')
    def validate_trials(cls, v):
        """Validate number of trials"""
        if v < 1:
            raise ValueError("trials_per_config must be at least 1")
        if v > 10:
            raise ValueError("trials_per_config should not exceed 10 for practical reasons")
        return v

    @property
    def dataset_choice_enum(self) -> DatasetChoice:
        """Convert string dataset choice to enum"""
        return DatasetChoice(self.dataset_choice)
    
    @property
    def dataset_config(self):
        """Get the configuration for the selected dataset"""
        return DATASET_CONFIGS[self.dataset_choice_enum]
    
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
        return f"{self.dataset_choice}_confidence_experiment_{self.name}"


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load and validate experiment configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ExperimentConfig(**config_data)


def list_available_configs(config_dir: str = "yaml/experiments") -> List[str]:
    """List all available experiment configuration files"""
    config_path = Path(config_dir)
    
    if not config_path.exists():
        return []
    
    return [f.stem for f in config_path.glob("*.yaml")]


def load_all_configs(config_dir: str = "yaml/experiments") -> List[ExperimentConfig]:
    """Load all experiment configurations from directory"""
    config_path = Path(config_dir)
    configs = []
    print(f"Loading experiment configurations from {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    
    for config_file in config_path.glob("*.yaml"):
        try:
            print(f"Loading config from {config_file}")
            config = load_experiment_config(str(config_file))
            configs.append(config)
        except Exception as e:
            print(f"Warning: Failed to load config {config_file}: {e}")
    
    return configs