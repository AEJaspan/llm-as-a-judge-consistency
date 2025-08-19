import pandas as pd
import numpy as np
import json
import asyncio
from experiment.classifier import LLMJudge
from experiment.confidence import ConfidenceExperiment
from config.base_models import DatasetChoice, DATASET_CONFIGS, ExperimentConfig
# Example usage and testing functions
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # Load environment variables from .env file
# Define the different confidence schema variants

def test_single_prediction():
    """Test a single prediction to verify the setup works"""
    judge = LLMJudge("gpt-4o-mini")  # Make sure to set your API key
    
    # Test with both datasets
    test_cases = {
        DatasetChoice.SST2: "This movie was absolutely fantastic! I loved every minute of it.",
        DatasetChoice.SMS_SPAM: "URGENT: You've won £1000! Text CLAIM to 88888 now!"
    }
    
    print("Testing single predictions...")
    for dataset_choice, test_text in test_cases.items():
        config = DATASET_CONFIGS[dataset_choice]
        print(f"\n{dataset_choice.value.upper()} Dataset:")
        print(f"Text: {test_text}")
        print(f"Task: {config['task_description']}")
        
        for conf_type in ["float", "categorical", "integer"]:
            try:
                result = judge.classify_with_confidence_sync(
                    test_text, conf_type, config['task_description']
                )
                print(f"  {conf_type}: {result}")
            except Exception as e:
                print(f"  Error with {conf_type}: {e}")

def run_sst2_experiment():
    """Run experiment on SST-2 dataset"""
    config = ExperimentConfig(
        dataset_choice=DatasetChoice.SST2,
        sample_size=20,  # Smaller for testing
        models=["gpt-4o-mini", "gpt-4o"],
        confidence_types=["float", "categorical", "integer"]
    )
    
    experiment = ConfidenceExperiment(config)
    return experiment

def run_sms_spam_experiment():
    """Run experiment on SMS Spam dataset"""
    config = ExperimentConfig(
        dataset_choice=DatasetChoice.SMS_SPAM,
        sample_size=20,  # Smaller for testing
        models=["gpt-4o-mini", "gpt-4o"],
        confidence_types=["float", "categorical", "integer"]
    )
    
    experiment = ConfidenceExperiment(config)
    return experiment

async def main(config: ExperimentConfig):   
    config = ExperimentConfig(
        dataset_choice=DatasetChoice.SMS_SPAM,
        sample_size=50,
        models = ["gpt-4o-mini", "gpt-4o"],
        # ["gpt-3.5-turbo", "claude-3-5-sonnet-20241022"],
        confidence_types=["float", "categorical", "integer"]
    )
    
    # Run experiment
    experiment = ConfidenceExperiment(config)
    results_df = await experiment.run_experiment()
    
    # Analyze results
    analyses = experiment.analyze_results(results_df)
    
    # Create visualizations
    experiment.create_visualizations(results_df, analyses)
    
    # Save results
    results_df.to_csv('llm_confidence_experiment_results.csv', index=False)
    def find_invalid_keys(data, path=""):
        """Recursively finds and prints invalid keys in a dictionary."""
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, (str, int, float, bool, type(None))):
                    print(f"Invalid key type found: {type(key)}")
                    print(f"Problematic key at path '{path}': {key}")
                    # You can stop here or continue to find all invalid keys
                    return
                # Recursively check the value if it's a dictionary
                find_invalid_keys(value, path=f"{path}['{key}']")

    # Insert this code right before your json.dump call
    print("Now performing a deep check for non-serializable keys...")
    find_invalid_keys(analyses)
    print("Deep check passed. Saving now...")
    with open('llm_confidence_analyses.json', 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    
    return results_df, analyses

# Statistical significance testing functions
def test_confidence_format_significance(results_df):
    """Test if different confidence formats lead to significantly different consistency"""
    from scipy.stats import kruskal, mannwhitneyu
    
    # Calculate consistency metric for each text-model-format combination
    consistency_data = []
    for (model, text_id), group in results_df.groupby(['model', 'text_id']):
        for conf_type in group['confidence_type'].unique():
            type_group = group[group['confidence_type'] == conf_type]
            if len(type_group) > 1:
                confidences = type_group['normalized_confidence'].values
                cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else 0
                consistency_data.append({
                    'model': model,
                    'text_id': text_id,
                    'confidence_type': conf_type,
                    'consistency': cv
                })
    
    consistency_df = pd.DataFrame(consistency_data)
    
    # Kruskal-Wallis test for differences between confidence formats
    groups = [group['consistency'].values for name, group in consistency_df.groupby('confidence_type')]
    h_stat, p_value = kruskal(*groups)
    
    print(f"Kruskal-Wallis test for confidence format differences:")
    print(f"H-statistic: {h_stat:.3f}, p-value: {p_value:.3f}")
    
    # Pairwise comparisons
    conf_types = consistency_df['confidence_type'].unique()
    for i, type1 in enumerate(conf_types):
        for type2 in conf_types[i+1:]:
            group1 = consistency_df[consistency_df['confidence_type'] == type1]['consistency'].values
            group2 = consistency_df[consistency_df['confidence_type'] == type2]['consistency'].values
            
            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            print(f"{type1} vs {type2}: U-statistic: {u_stat:.3f}, p-value: {p_val:.3f}")




# Additional utility functions for dataset comparison
def compare_dataset_results(sst2_results_file: str, spam_results_file: str):
    """Compare results between SST2 and SMS spam experiments"""
    sst2_df = pd.read_csv(sst2_results_file)
    spam_df = pd.read_csv(spam_results_file)
    
    print("=== DATASET COMPARISON ===")
    
    # Basic stats comparison
    print("\nBasic Statistics:")
    for name, df in [("SST2", sst2_df), ("SMS_SPAM", spam_df)]:
        accuracy = df['correct'].mean()
        mean_confidence = df['normalized_confidence'].mean()
        print(f"{name}: Accuracy={accuracy:.3f}, Mean Confidence={mean_confidence:.3f}")
    
    # Consistency comparison
    print("\nConsistency (CV by confidence type):")
    for df, name in [(sst2_df, "SST2"), (spam_df, "SMS_SPAM")]:
        print(f"\n{name}:")
        for conf_type in df['confidence_type'].unique():
            type_data = df[df['confidence_type'] == conf_type]
            # Calculate consistency per text
            consistency_scores = []
            for text_id in type_data['text_id'].unique():
                text_data = type_data[type_data['text_id'] == text_id]
                if len(text_data) > 1:
                    cv = np.std(text_data['normalized_confidence']) / np.mean(text_data['normalized_confidence'])
                    consistency_scores.append(cv)
            
            if consistency_scores:
                print(f"  {conf_type}: CV = {np.mean(consistency_scores):.3f}")

def run_both_experiments():
    """Convenience function to run both experiments sequentially"""
    
    print("=== RUNNING SST2 EXPERIMENT ===")
    sst2_config = ExperimentConfig(
        dataset_choice=DatasetChoice.SST2,
        sample_size=50,
        models=["gpt-4o-mini", "gpt-4o"],
        confidence_types=["float", "categorical", "integer"]
    )
    
    sst2_experiment = ConfidenceExperiment(sst2_config)
    sst2_results = asyncio.run(sst2_experiment.run_experiment())
    sst2_analyses = sst2_experiment.analyze_results(sst2_results)
    sst2_experiment.create_visualizations(sst2_results, sst2_analyses)
    
    # Save SST2 results
    sst2_results.to_csv('llm_confidence_experiment_results_sst2.csv', index=False)
    
    print("\n" + "="*50)
    print("=== RUNNING SMS SPAM EXPERIMENT ===")
    
    spam_config = ExperimentConfig(
        dataset_choice=DatasetChoice.SMS_SPAM,
        sample_size=50,
        models=["gpt-4o-mini", "gpt-4o"],
        confidence_types=["float", "categorical", "integer"]
    )
    
    spam_experiment = ConfidenceExperiment(spam_config)
    spam_results = asyncio.run(spam_experiment.run_experiment())
    spam_analyses = spam_experiment.analyze_results(spam_results)
    spam_experiment.create_visualizations(spam_results, spam_analyses)
    
    # Save spam results
    spam_results.to_csv('llm_confidence_experiment_results_sms_spam.csv', index=False)
    
    print("\n" + "="*50)
    print("=== COMPARING RESULTS ===")
    
    compare_dataset_results(
        'llm_confidence_experiment_results_sst2.csv',
        'llm_confidence_experiment_results_sms_spam.csv'
    )
    
    return sst2_results, spam_results, sst2_analyses, spam_analyses



if __name__ == "__main__":
    # Test single prediction first
    print("=== Testing Single Prediction ===")
    test_single_prediction()
    
    print("\n=== Running Full Experiment ===")
    # Run the full experiment
    sst2_results, spam_results, sst2_analyses, spam_analyses = run_both_experiments()
    # results_df, analyses = asyncio.run(main())

    # for results_df, analyses in [(results_df, analyses)]:
    for results_df, analyses in [(sst2_results, sst2_analyses), (spam_results, spam_analyses)]:
        # Additional statistical testing
        test_confidence_format_significance(results_df)
        
        dataset_name = results_df['dataset'].iloc[0] if 'dataset' in results_df.columns else 'unknown'
        print(f"\nExperiment completed for {dataset_name.upper()} dataset!")
        print("Generated files:")
        print(f"- llm_confidence_experiment_results_{dataset_name}.csv")
        print(f"- llm_confidence_analyses_{dataset_name}.json")
        print(f"- llm_confidence_experiment_results_{dataset_name}.png")
        
        print(f"\n=== QUICK COMPARISON GUIDE ===")
        print("To compare datasets, run the experiment twice:")
        print("1. First with DatasetChoice.SST2")
        print("2. Then with DatasetChoice.SMS_SPAM")
        print("3. Compare the generated CSV files and plots")
