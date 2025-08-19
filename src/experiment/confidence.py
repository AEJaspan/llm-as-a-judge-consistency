import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Dict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss
import asyncio
from tqdm import tqdm
from config.base_models import (
    DatasetChoice,
    ExperimentConfig,
    CATEGORICAL_TO_FLOAT
)
from config.constants import ExperimentConstants

from experiment.classifier import LLMJudge

class ConfidenceExperiment:
    def __init__(self, config: ExperimentConfig, max_concurrent_requests: int = 5):
        self.config = config
        self.results = []
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
    async def _rate_limited_llm_call(self, judge: LLMJudge, text: str, conf_type: str, task_description: str):
        """Make an LLM call with rate limiting"""
        async with self.semaphore:
            return await judge.classify_with_confidence(text, conf_type, task_description)
        
    def load_dataset(self):
        """Load and prepare the dataset"""
        config = self.config.dataset_config
        
        if self.config.dataset_choice == DatasetChoice.SST2:
            dataset = load_dataset(self.config.dataset_name, self.config.dataset_config_name)
            # Use validation split for SST2 as it's a good size for testing
            split_name = "validation" if "validation" in dataset else "train"
            df = pd.DataFrame(dataset[split_name])
            
        elif self.config.dataset_choice == DatasetChoice.SMS_SPAM:
            dataset = load_dataset(self.config.dataset_name, self.config.dataset_config_name)
            # SMS spam comes as one split, we'll use it directly
            df = pd.DataFrame(dataset['train'])
            
            # For SMS spam, do stratified sampling to maintain class balance
            if len(df) > self.config.sample_size:
                from sklearn.model_selection import train_test_split
                df, _ = train_test_split(
                    df, 
                    train_size=self.config.sample_size, 
                    random_state=42, 
                    stratify=df[self.config.label_column]
                )
        
        # General sampling if needed (for SST2 or if SMS doesn't need stratification)
        if len(df) > self.config.sample_size and self.config.dataset_choice == DatasetChoice.SST2:
            df = df.sample(n=self.config.sample_size, random_state=42)
            
        print(f"Loaded {self.config.dataset_choice.value} dataset:")
        print(f"  Description: {config['description']}")
        print(f"  Total samples: {len(df)}")
        print(f"  Text column: '{self.config.text_column}'")
        print(f"  Label column: '{self.config.label_column}'")
        print(f"  Label distribution: {df[self.config.label_column].value_counts().to_dict()}")
        print(f"  Label meanings: {config['label_meanings']}")
        
        return df[[self.config.text_column, self.config.label_column]]
    
    async def run_experiment(self):
        """Run the full experiment with concurrent LLM calls"""
        print(f"=== Starting {self.config.dataset_choice.value.upper()} Confidence Experiment ===")
        print("Loading dataset...")
        df = self.load_dataset()
        
        models = self.config.models or ['gpt-4o-mini']
        # ["gpt-3.5-turbo", "claude-3-5-sonnet-20241022"]
        confidence_types = self.config.confidence_types or ["float", "categorical", "integer"]
        
        # Create all LLM call tasks upfront
        all_tasks = []
        task_metadata = []
        
        print("Creating LLM tasks...")
        for model_name in models:
            print(f"  Preparing tasks for model: {model_name}")
            judge = LLMJudge(model_name)
            
            for conf_type in confidence_types:
                for trial in range(3):  # 3 trials per configuration
                    for idx, row in df.iterrows():
                        text = row[self.config.text_column]
                        true_label = bool(row[self.config.label_column])
                        
                        # Create the async task with rate limiting
                        task = self._rate_limited_llm_call(
                            judge, text, conf_type, self.config.task_description
                        )
                        all_tasks.append(task)
                        
                        # Store metadata for this task
                        task_metadata.append({
                            "dataset": self.config.dataset_choice.value,
                            "model": model_name,
                            "confidence_type": conf_type,
                            "trial": trial,
                            "text_id": idx,
                            "true_label": true_label,
                            "text": text  # Store for debugging if needed
                        })
        
        print(f"Created {len(all_tasks)} total LLM tasks")
        print("Executing all tasks concurrently...")
        
        # Execute all tasks concurrently with progress tracking
        results = []
        batch_size = ExperimentConstants.DEFAULT_BATCH_SIZE
        counter = tqdm(total=len(all_tasks), desc="Processing tasks", unit=" task")
        for i in range(0, len(all_tasks), batch_size):
            batch_tasks = all_tasks[i:i + batch_size]
            batch_metadata = task_metadata[i:i + batch_size]
            counter.update(len(batch_tasks))
            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for task_result, metadata in zip(batch_results, batch_metadata):
                if isinstance(task_result, Exception):
                    print(f"    Error in task: {task_result}")
                    # Use fallback result for failed tasks
                    task_result = {
                        "classification": None, 
                        "confidence": None 
                    }
                
                # Normalize confidence to 0-1 scale for analysis
                normalized_confidence = self._normalize_confidence(
                    task_result["confidence"], metadata["confidence_type"]
                )
                
                result_entry = {
                    "dataset": metadata["dataset"],
                    "model": metadata["model"],
                    "confidence_type": metadata["confidence_type"],
                    "trial": metadata["trial"],
                    "text_id": metadata["text_id"],
                    "true_label": metadata["true_label"],
                    "predicted_label": task_result["classification"],
                    "raw_confidence": task_result["confidence"],
                    "normalized_confidence": normalized_confidence,
                    "correct": metadata["true_label"] == task_result["classification"]
                }
                results.append(result_entry)
            
            # Add small delay between batches to be respectful to APIs
            await asyncio.sleep(1)
        
        print(f"Completed all {len(results)} tasks!")
        self.results = results
        return pd.DataFrame(self.results)
    
    def _normalize_confidence(self, confidence, conf_type):
        """Normalize different confidence formats to 0-1 scale"""
        if conf_type == "float":
            return confidence
        elif conf_type == "categorical":
            return CATEGORICAL_TO_FLOAT[confidence]
        elif conf_type == "integer":
            return confidence / 5.0
    
    def analyze_results(self, results_df: pd.DataFrame):
        """Perform statistical analysis of the results"""
        analyses = {}
        dataset_name = results_df['dataset'].iloc[0] if 'dataset' in results_df.columns else 'unknown'
        
        # 1. Consistency Analysis (across trials)
        print(f"=== CONSISTENCY ANALYSIS ({dataset_name.upper()}) ===")
        consistency_results = self._analyze_consistency(results_df)
        analyses["consistency"] = consistency_results
        
        # 2. Calibration Analysis
        print(f"\n=== CALIBRATION ANALYSIS ({dataset_name.upper()}) ===")
        calibration_results = self._analyze_calibration(results_df)
        analyses["calibration"] = calibration_results
        
        # 3. Confidence Distribution Analysis
        print(f"\n=== CONFIDENCE DISTRIBUTION ANALYSIS ({dataset_name.upper()}) ===")
        distribution_results = self._analyze_distributions(results_df)
        analyses["distributions"] = distribution_results
        
        # 4. Model Comparison
        print(f"\n=== MODEL COMPARISON ({dataset_name.upper()}) ===")
        model_comparison = self._compare_models(results_df)
        analyses["model_comparison"] = model_comparison
        
        # Add dataset metadata to analyses
        analyses["dataset_info"] = {
            "dataset_name": dataset_name,
            "total_samples": len(results_df['text_id'].unique()),
            "models_tested": results_df['model'].unique().tolist(),
            "confidence_types": results_df['confidence_type'].unique().tolist(),
            "trials_per_config": results_df['trial'].max() + 1
        }
        
        return analyses
    
    def _analyze_consistency(self, df: pd.DataFrame):
        """Analyze consistency of confidence scores across trials"""
        consistency_results = {}
        
        for (model, conf_type), group in df.groupby(['model', 'confidence_type']):
            # Calculate coefficient of variation for each text across trials
            text_consistency = []
            
            for text_id, text_group in group.groupby('text_id'):
                confidences = text_group['normalized_confidence'].values
                if len(confidences) > 1:
                    cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else np.inf
                    text_consistency.append(cv)
            
            consistency_results[f"{model}---{conf_type}"] = {
                "mean_cv": np.mean(text_consistency),
                "std_cv": np.std(text_consistency),
                "median_cv": np.median(text_consistency)
            }
            
            print(f"{model} - {conf_type}: Mean CV = {np.mean(text_consistency):.3f}")
        
        return consistency_results
    
    def _analyze_calibration(self, df: pd.DataFrame):
        """Analyze how well confidence scores match actual accuracy"""
        calibration_results = {}
        
        for (model, conf_type), group in df.groupby(['model', 'confidence_type']):
            # Calculate Brier score and reliability
            confidences = group['normalized_confidence'].values
            correct = group['correct'].astype(int).values
            
            brier_score = brier_score_loss(correct, confidences)
            
            # Bin confidences and calculate calibration curve
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_curve = []
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correct[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    calibration_curve.append({
                        'bin_lower': bin_lower,
                        'bin_upper': bin_upper,
                        'accuracy': accuracy_in_bin,
                        'confidence': avg_confidence_in_bin,
                        'count': in_bin.sum()
                    })
            
            calibration_results[f"{model}---{conf_type}"] = {
                "brier_score": brier_score,
                "calibration_curve": calibration_curve
            }
            
            print(f"{model} - {conf_type}: Brier Score = {brier_score:.3f}")
        
        return calibration_results
    
    def _analyze_distributions(self, df: pd.DataFrame):
        """Analyze the distribution of confidence scores"""
        distribution_results = {}
        
        for (model, conf_type), group in df.groupby(['model', 'confidence_type']):
            confidences = group['normalized_confidence'].values
            
            distribution_results[f"{model}---{conf_type}"] = {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "skewness": stats.skew(confidences),
                "kurtosis": stats.kurtosis(confidences),
                "entropy": stats.entropy(np.histogram(confidences, bins=10)[0] + 1e-10)
            }
        
        return distribution_results
    
    def _compare_models(self, df: pd.DataFrame):
        """Statistical comparison between models and confidence types"""
        comparison_results = {}
        
        # ANOVA for consistency differences
        consistency_data = []
        for (model, conf_type), group in df.groupby(['model', 'confidence_type']):
            for text_id, text_group in group.groupby('text_id'):
                confidences = text_group['normalized_confidence'].values
                if len(confidences) > 1:
                    cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else 0
                    consistency_data.append({
                        'model': model,
                        'confidence_type': conf_type,
                        'consistency': cv
                    })
        
        consistency_df = pd.DataFrame(consistency_data)
        
        # Statistical tests
        if len(consistency_df) > 0:
            # Compare confidence types
            conf_type_groups = [group['consistency'].values 
                              for name, group in consistency_df.groupby('confidence_type')]
            if len(conf_type_groups) > 2:
                f_stat, p_value = stats.f_oneway(*conf_type_groups)
                comparison_results['confidence_type_anova'] = {'f_stat': f_stat, 'p_value': p_value}
            
            # Compare models
            model_groups = [group['consistency'].values 
                          for name, group in consistency_df.groupby('model')]
            if len(model_groups) > 2:
                f_stat, p_value = stats.f_oneway(*model_groups)
                comparison_results['model_anova'] = {'f_stat': f_stat, 'p_value': p_value}
        
        return comparison_results
    
    def create_visualizations(self, results_df: pd.DataFrame, analyses: Dict):
        """Create visualizations of the results"""
        dataset_name = results_df['dataset'].iloc[0] if 'dataset' in results_df.columns else 'unknown'
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'LLM Confidence Experiment Results - {dataset_name.upper()} Dataset', 
                     fontsize=16, fontweight='bold')
        
        # 1. Consistency comparison
        consistency_data = []
        for key, metrics in analyses['consistency'].items():
            print(key)
            model, conf_type = key.split('---')
            consistency_data.append({
                'Model': model,
                'Confidence Type': conf_type,
                'Mean CV': metrics['mean_cv']
            })
        
        consistency_df = pd.DataFrame(consistency_data)
        sns.barplot(data=consistency_df, x='Confidence Type', y='Mean CV', hue='Model', ax=axes[0,0])
        axes[0,0].set_title('Consistency Comparison (Lower is Better)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Brier Score comparison
        brier_data = []
        for key, metrics in analyses['calibration'].items():
            model, conf_type = key.split('---')
            brier_data.append({
                'Model': model,
                'Confidence Type': conf_type,
                'Brier Score': metrics['brier_score']
            })
        
        brier_df = pd.DataFrame(brier_data)
        sns.barplot(data=brier_df, x='Confidence Type', y='Brier Score', hue='Model', ax=axes[0,1])
        axes[0,1].set_title('Calibration Quality (Lower is Better)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Confidence distributions
        sns.boxplot(data=results_df, x='confidence_type', y='normalized_confidence', 
                   hue='model', ax=axes[0,2])
        axes[0,2].set_title('Confidence Score Distributions')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Accuracy by confidence level
        results_df['confidence_bin'] = pd.cut(results_df['normalized_confidence'], 
                                            bins=5, labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
        acc_by_conf = results_df.groupby(['confidence_type', 'confidence_bin'])['correct'].mean().reset_index()
        sns.lineplot(data=acc_by_conf, x='confidence_bin', y='correct', 
                    hue='confidence_type', marker='o', ax=axes[1,0])
        axes[1,0].set_title('Accuracy vs Confidence Level')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Model performance comparison
        model_perf = results_df.groupby(['model', 'confidence_type']).agg({
            'correct': 'mean',
            'normalized_confidence': 'mean'
        }).reset_index()
        
        sns.scatterplot(data=model_perf, x='normalized_confidence', y='correct', 
                       hue='model', style='confidence_type', s=100, ax=axes[1,1])
        axes[1,1].set_title('Accuracy vs Mean Confidence')
        axes[1,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Perfect calibration line
        
        # 6. Confidence type comparison heatmap
        pivot_data = consistency_df.pivot(index='Model', columns='Confidence Type', values='Mean CV')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,2])
        axes[1,2].set_title('Consistency Heatmap')
        
        plt.tight_layout()
        
        # Create filename with dataset name
        filename = f'llm_confidence_experiment_results_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filename