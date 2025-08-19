from enum import Enum


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ExperimentConstants:
    DEFAULT_BATCH_SIZE = 20
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_TIMEOUT = 30
    MIN_SAMPLE_SIZE = 10
    MAX_SAMPLE_SIZE = 100000


class APIConstants:
    OPENAI_RATE_LIMIT = 50  # requests per minute
    ANTHROPIC_RATE_LIMIT = 30
    DEFAULT_TEMPERATURE = 0.0
