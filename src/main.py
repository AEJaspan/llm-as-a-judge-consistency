import pandas as pd
import numpy as np
import json
import asyncio
import argparse
from typing import List, Dict, Tuple
from experiment.classifier import LLMJudge
from experiment.confidence import ConfidenceExperiment
from config.base_models import (
    ExperimentConfig,
    load_experiment_config,
    list_available_configs,
    load_all_configs
)
from dotenv import load_dotenv, find_dotenv
from config.constants import ExperimentConstants, APIConstants
from scipy.stats import kruskal, mannwhitneyu

load_dotenv(find_dotenv())


def test_single_prediction(config: ExperimentConfig) -> None:
    """Test a single prediction to verify the setup works"""
    print(f"=== Testing Single Prediction for {config.name} ===")
    
    judge = LLMJudge(
        config.models[0],  # Use first model for testing
        temperature=APIConstants.DEFAULT_TEMPERATURE,
        max_retries=ExperimentConstants.DEFAULT_MAX_RETRIES,
    )
    
    # Test cases for different datasets
    test_cases = {
        "sst2": "This movie was absolutely fantastic! I loved every minute of it.",
        "sms_spam": "URGENT: You've won £1000! Text CLAIM to 88888 now!",
    }
    
    test_text = test_cases.get(config.dataset_choice, "Test message for classification.")
    dataset_config = config.dataset_config
    
    print(f"Dataset: {config.dataset_choice.upper()}")
    print(f"Text: {test_text}")
    print(f"Task: {dataset_config['task_description']}")
    print(f"Model: {config.models[0]}")
    
    for conf_type in config.confidence_types:
        try:
            result = judge.classify_with_confidence_sync(
                test_text, conf_type, dataset_config["task_description"]
            )
            print(f"  {conf_type}: {result}")
        except Exception as e:
            print(f"  Error with {conf_type}: {e}")


async def run_single_experiment(config: ExperimentConfig) -> Tuple[pd.DataFrame, Dict]:
    """Run a single experiment with the given configuration"""
    print(f"\n=== Running Experiment: {config.name} ===")
    print(f"Description: {config.description}")
    print(f"Dataset: {config.dataset_choice}")
    print(f"Sample size: {config.sample_size}")
    print(f"Models: {config.models}")
    print(f"Confidence types: {config.confidence_types}")
    print(f"Trials per config: {config.trials_per_config}")
    
    # Create and run experiment
    experiment = ConfidenceExperiment(config, max_concurrent_requests=config.max_concurrent_requests)
    results_df = await experiment.run_experiment()
    
    # Analyze results
    analyses = experiment.analyze_results(results_df)
    
    # Create visualizations
    plot_filename = experiment.create_visualizations(results_df, analyses)
    
    # Save results with experiment name
    results_filename = f"llm_confidence_experiment_results_{config.name}.csv"
    analyses_filename = f"llm_confidence_analyses_{config.name}.json"
    
    results_df.to_csv(results_filename, index=False)
    with open(analyses_filename, "w") as f:
        json.dump(analyses, f, indent=2, default=str)
    
    print(f" Experiment '{config.name}' completed!")
    print(f"  Results: {results_filename}")
    print(f"  Analysis: {analyses_filename}")
    print(f"  Plot: {plot_filename}")
    
    return results_df, analyses


async def run_multiple_experiments(configs: List[ExperimentConfig]) -> List[Tuple[pd.DataFrame, Dict, str]]:
    """Run multiple experiments sequentially"""
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i}/{len(configs)}: {config.name}")
        print(f"{'='*60}")
        
        try:
            results_df, analyses = await run_single_experiment(config)
            results.append((results_df, analyses, config.name))
        except Exception as e:
            print(f" Experiment '{config.name}' failed: {e}")
            continue
    
    return results


def test_confidence_format_significance(results_df: pd.DataFrame, experiment_name: str) -> None:
    """Test if different confidence formats lead to significantly different consistency"""
    print(f"\n=== Statistical Significance Analysis: {experiment_name} ===")
    
    # Calculate consistency metric for each text-model-format combination
    consistency_data = []
    for (model, text_id), group in results_df.groupby(["model", "text_id"]):
        for conf_type in group["confidence_type"].unique():
            type_group = group[group["confidence_type"] == conf_type]
            if len(type_group) > 1:
                confidences = type_group["normalized_confidence"].values
                cv = (
                    np.std(confidences) / np.mean(confidences)
                    if np.mean(confidences) > 0
                    else 0
                )
                consistency_data.append(
                    {
                        "model": model,
                        "text_id": text_id,
                        "confidence_type": conf_type,
                        "consistency": cv,
                    }
                )

    if not consistency_data:
        print(" No consistency data available for statistical testing")
        return

    consistency_df = pd.DataFrame(consistency_data)

    # Kruskal-Wallis test for differences between confidence formats
    groups = [
        group["consistency"].values
        for name, group in consistency_df.groupby("confidence_type")
    ]
    
    if len(groups) > 2:
        h_stat, p_value = kruskal(*groups)
        print(f"Kruskal-Wallis test for confidence format differences:")
        print(f"  H-statistic: {h_stat:.3f}, p-value: {p_value:.3f}")
        
        # Pairwise comparisons
        conf_types = consistency_df["confidence_type"].unique()
        print("\nPairwise comparisons:")
        for i, type1 in enumerate(conf_types):
            for type2 in conf_types[i + 1 :]:
                group1 = consistency_df[consistency_df["confidence_type"] == type1][
                    "consistency"
                ].values
                group2 = consistency_df[consistency_df["confidence_type"] == type2][
                    "consistency"
                ].values

                u_stat, p_val = mannwhitneyu(group1, group2, alternative="two-sided")
                significance = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {type1} vs {type2}: U={u_stat:.3f}, p={p_val:.3f} {significance}")


def compare_experiment_results(experiment_results: List[Tuple[pd.DataFrame, Dict, str]]) -> None:
    """Compare results across multiple experiments"""
    if len(experiment_results) < 2:
        print(" Need at least 2 experiments for comparison")
        return
    
    print(f"\n{'='*60}")
    print("CROSS-EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = []
    
    for results_df, analyses, name in experiment_results:
        dataset_name = results_df["dataset"].iloc[0] if "dataset" in results_df.columns else "unknown"
        accuracy = results_df["correct"].mean()
        mean_confidence = results_df["normalized_confidence"].mean()
        
        # Calculate overall consistency
        consistency_scores = []
        for (model, conf_type), group in results_df.groupby(["model", "confidence_type"]):
            for text_id, text_group in group.groupby("text_id"):
                confidences = text_group["normalized_confidence"].values
                if len(confidences) > 1:
                    cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else 0
                    consistency_scores.append(cv)
        
        mean_consistency = np.mean(consistency_scores) if consistency_scores else float('nan')
        
        comparison_data.append({
            "experiment": name,
            "dataset": dataset_name,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "mean_consistency_cv": mean_consistency,
            "sample_size": len(results_df["text_id"].unique()),
            "total_predictions": len(results_df)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nExperiment Summary:")
    print(comparison_df.round(3).to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv("experiment_comparison_results.csv", index=False)
    print("\n✓ Comparison saved to experiment_comparison_results.csv")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Confidence Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --list                           # List available configs
  python main.py --config quick_test              # Run single experiment
  python main.py --config sst2_full sms_spam_full # Run multiple experiments
  python main.py --all                            # Run all available experiments
  python main.py --test-only --config quick_test  # Just test setup
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        nargs="+",
        help="Configuration file(s) to run (without .yaml extension)"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all available experiment configurations"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiment configurations"
    )
    
    parser.add_argument(
        "--test-only", "-t",
        action="store_true",
        help="Only run single prediction test, skip full experiment"
    )
    
    parser.add_argument(
        "--config-dir",
        default="yaml/experiments",
        help="Directory containing experiment configuration files"
    )
    
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistical significance testing"
    )
    
    return parser


async def main():
    """Main function to orchestrate experiment execution"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # List available configurations
    if args.list:
        configs = list_available_configs(args.config_dir)
        if configs:
            print("Available experiment configurations:")
            for config in configs:
                print(f"  - {config}")
        else:
            print(f"No configuration files found in {args.config_dir}")
        return
    
    # Load configurations
    try:
        if args.all:
            configs = load_all_configs(args.config_dir)
            if not configs:
                print(f" No valid configurations found in {args.config_dir}")
                return
        elif args.config:
            configs = []
            for config_name in args.config:
                config_path = f"{args.config_dir}/{config_name}.yaml"
                print(f"Loading config '{config_name}' from {config_path}")
                try:
                    config = load_experiment_config(config_path)
                    configs.append(config)
                except Exception as e:
                    print(f" Failed to load config '{config_name}': {e}")
        else:
            # Default: run quick test
            config_path = f"{args.config_dir}/quick_test.yaml"
            configs = [load_experiment_config(config_path)]
            
    except Exception as e:
        print(f" Error loading configurations: {e}")
        return
    
    if not configs:
        print(" No valid configurations to run")
        return
    
    print(f"Loaded {len(configs)} experiment configuration(s)")
    
    # Test setup for first configuration
    print("\n" + "="*60)
    print("SETUP VERIFICATION")
    print("="*60)
    test_single_prediction(configs[0])
    
    if args.test_only:
        print("\n Test completed successfully!")
        return
    
    # Run experiments
    print(f"\n{'='*60}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*60}")
    
    experiment_results = await run_multiple_experiments(configs)
    
    if not experiment_results:
        print(" No experiments completed successfully")
        return
    
    # Statistical analysis for each experiment
    if not args.no_stats:
        for results_df, analyses, name in experiment_results:
            test_confidence_format_significance(results_df, name)
    
    # Cross-experiment comparison
    if len(experiment_results) > 1:
        compare_experiment_results(experiment_results)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUITE COMPLETED")
    print(f"{'='*60}")
    print(f" Successfully completed {len(experiment_results)} experiment(s)")
    print(" Results, analyses, and plots have been generated")
    print(" Use 'make move-assets' to organize files into assets/ directory")


if __name__ == "__main__":
    asyncio.run(main())
