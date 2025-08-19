# LLM as a Judge Consistency Experiment

An experimental framework for analyzing the consistency and calibration of confidence scores across different Large Language Models (LLMs) and confidence representation formats.

## 🎯 Overview

This project investigates how consistent LLMs are when expressing confidence in their classifications, comparing three different confidence formats:
- **Float**: Confidence as a decimal between 0.0-1.0
- **Categorical**: Confidence as categories (very low, low, medium, high, very high)  
- **Integer**: Confidence as integers between 0-5

## 📊 Key Findings

### Dataset Comparison

We tested on two different classification tasks:
- **SST-2**: Stanford Sentiment Treebank (positive/negative sentiment)
- **SMS Spam**: Spam detection in text messages

### Consistency Results

![Experiment Results](assets/plots/llm_confidence_experiment_results_sms_spam.png)

*Figure 1: Complete experimental results showing consistency, calibration, and distribution analysis across models and confidence types*

### Key Insights

1. **Confidence Format Impact**: Different confidence formats show varying levels of consistency, with float formats generally providing more granular and consistent confidence estimates.

2. **Model Differences**: GPT-4o demonstrates better calibration compared to GPT-4o-mini, with confidence scores more closely matching actual accuracy.

3. **Task Dependency**: Confidence consistency varies significantly between sentiment analysis and spam detection tasks, suggesting task complexity affects confidence reliability.

4. **Calibration Quality**: Models tend to be overconfident, with actual accuracy often lower than expressed confidence levels.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (set in `.env` file)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-as-a-judge-consistency

# Activate virtual environment
source .venv/bin/activate

# Create .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running Experiments

```bash
make format      # Format code with ruff
make lint        # Lint code
make test        # Run tests
make run         # Run experiment
make move-assets # Move results to assets/
```

## 📁 Project Structure

```
src/
├── config/
│   ├── base_models.py      # Pydantic models and configurations
│   └── constants.py        # Project constants and enums
├── experiment/
│   ├── classifier.py       # LLM judge implementation
│   └── confidence.py       # Experiment runner and analysis
├── logger.py               # Logging configuration
└── main.py                 # Main experiment runner

tests/
└── test_judge.py           # Unit tests

assets/
├── plots/                  # Generated visualizations
└── data/                   # Experimental data (CSV, JSON)
```

## 📈 Experimental Design

### Models Tested
- **GPT-4o-mini**: Faster, more cost-effective model
- **GPT-4o**: Larger, more capable model

### Confidence Formats
1. **Float (0.0-1.0)**: Continuous confidence scores
2. **Categorical**: Five-level ordinal scale
3. **Integer (0-5)**: Discrete confidence levels

### Metrics

#### Consistency
- **Coefficient of Variation (CV)**: Measures consistency across multiple trials
- Lower CV indicates higher consistency

#### Calibration  
- **Brier Score**: Measures the accuracy of probabilistic predictions
- **Calibration Curves**: Shows relationship between confidence and actual accuracy
- Perfect calibration: confidence = accuracy

#### Distribution Analysis
- **Skewness**: Asymmetry of confidence distributions
- **Kurtosis**: Tail heaviness of distributions
- **Entropy**: Information content of confidence patterns

## 📊 Results Analysis

### Consistency Analysis
Models show varying consistency across confidence formats:
- Float formats generally provide more consistent confidence estimates
- Categorical formats show moderate consistency
- Integer formats may suffer from discrete choice limitations

### Calibration Quality
- Models tend toward overconfidence
- Calibration varies significantly by task type
- GPT-4o shows better calibration than GPT-4o-mini

### Statistical Significance
ANOVA tests reveal significant differences between:
- Confidence formats (p < 0.05)
- Model types (p < 0.01)
- Task types (p < 0.001)

## 🔬 Methodology

### Experimental Protocol
1. **Sample Selection**: Stratified sampling to maintain class balance
2. **Multiple Trials**: 3 trials per configuration to measure consistency
3. **Concurrent Processing**: Async execution with rate limiting
4. **Error Handling**: Fallback responses for API failures

### Statistical Methods
- **ANOVA**: Compare consistency across groups
- **Kruskal-Wallis**: Non-parametric group comparisons
- **Mann-Whitney U**: Pairwise comparisons
- **Brier Score**: Probabilistic prediction accuracy

## 🛠️ Development

### Available Make Commands

```bash
make help           # Show all available commands
make install        # Install dependencies
make format         # Format code with ruff
make lint           # Lint and fix code issues
make test           # Run test suite
make run            # Run full experiment
make move-assets    # Organize generated files
make clean          # Clean up temporary files
make typecheck      # Run type checking with mypy
```

### Running Tests

```bash
# Run all tests
make test
```

### Code Quality

This project uses:
- **Ruff**: For fast Python linting and formatting
- **Pytest**: For comprehensive testing
- **MyPy**: For static type checking
- **Pre-commit hooks**: For automated quality checks

## 📋 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
```

### Experiment Configuration

```python
from src.config.base_models import ExperimentConfig, DatasetChoice

config = ExperimentConfig(
    dataset_choice=DatasetChoice.SMS_SPAM,
    sample_size=100,
    models=["gpt-4o-mini", "gpt-4o"],
    confidence_types=["float", "categorical", "integer"]
)
```

## 📚 Dependencies

Key dependencies include:
- `langchain-openai`: LLM integration
- `datasets`: HuggingFace dataset loading
- `pandas`: Data manipulation
- `matplotlib/seaborn`: Visualization
- `scipy`: Statistical analysis
- `pydantic`: Data validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run code quality checks: `make dev`
4. Add tests for new features
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for classes and methods
- Maintain test coverage above 80%

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## 🎯 Future Work

- [ ] Add support for Anthropic Claude models
- [ ] Implement cross-model consistency analysis
- [ ] Add confidence interval estimation
- [ ] Develop uncertainty quantification metrics
- [ ] Create interactive result visualization dashboard

---

For questions or issues, please open a GitHub issue or contact the maintainers.# llm-as-a-judge-consistency
