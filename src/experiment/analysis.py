"""
Additional analysis functions to enhance the confidence calibration research.
Add these methods to your ConfidenceExperiment class or as standalone functions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

def create_calibration_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a detailed calibration table showing expected vs actual accuracy
    for different confidence bins.
    """
    calibration_data = []
    
    for (model, conf_type), group in results_df.groupby(['model', 'confidence_type']):
        # Create confidence bins
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        group['conf_bin'] = pd.cut(group['normalized_confidence'], 
                                   bins=bins, labels=bin_labels, include_lowest=True)
        
        for bin_label in bin_labels:
            bin_data = group[group['conf_bin'] == bin_label]
            if len(bin_data) > 0:
                calibration_data.append({
                    'Model': model,
                    'Confidence Type': conf_type,
                    'Confidence Bin': bin_label,
                    'Mean Confidence': bin_data['normalized_confidence'].mean(),
                    'Actual Accuracy': bin_data['correct'].mean(),
                    'Sample Count': len(bin_data),
                    'Calibration Error': abs(bin_data['normalized_confidence'].mean() - 
                                            bin_data['correct'].mean())
                })
    
    calib_df = pd.DataFrame(calibration_data)
    
    # Print formatted table
    print("\n=== DETAILED CALIBRATION TABLE ===")
    for (model, conf_type), group in calib_df.groupby(['Model', 'Confidence Type']):
        print(f"\n{model} - {conf_type}:")
        print(group[['Confidence Bin', 'Mean Confidence', 'Actual Accuracy', 
                    'Sample Count', 'Calibration Error']].round(3).to_string(index=False))
    
    return calib_df


def calculate_ece_mce(results_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    for each model-confidence type combination.
    """
    ece_mce_results = []
    
    for (model, conf_type), group in results_df.groupby(['model', 'confidence_type']):
        confidences = group['normalized_confidence'].values
        accuracies = group['correct'].values
        
        # Calculate ECE and MCE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
        
        ece_mce_results.append({
            'Model': model,
            'Confidence Type': conf_type,
            'ECE': ece,
            'MCE': mce,
            'Mean Confidence': confidences.mean(),
            'Accuracy': accuracies.mean()
        })
    
    ece_df = pd.DataFrame(ece_mce_results)
    
    print("\n=== CALIBRATION METRICS SUMMARY ===")
    print(ece_df.round(3).to_string(index=False))
    
    return ece_df


def create_reliability_diagram(results_df: pd.DataFrame, save_path: str = None):
    """
    Create reliability diagrams (calibration plots) for each model and confidence type.
    """
    n_models = results_df['model'].nunique()
    n_conf_types = results_df['confidence_type'].nunique()
    
    fig, axes = plt.subplots(n_models, n_conf_types, figsize=(15, 10))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_conf_types == 1:
        axes = axes.reshape(-1, 1)
    
    dataset_name = results_df['dataset'].iloc[0] if 'dataset' in results_df.columns else "unknown"
    fig.suptitle(f'Reliability Diagrams - {dataset_name.upper()} Dataset', fontsize=16)
    
    models = sorted(results_df['model'].unique())
    conf_types = sorted(results_df['confidence_type'].unique())
    
    for i, model in enumerate(models):
        for j, conf_type in enumerate(conf_types):
            ax = axes[i, j] if n_models > 1 else axes[j]
            
            # Get data for this combination
            mask = (results_df['model'] == model) & (results_df['confidence_type'] == conf_type)
            data = results_df[mask]
            
            if len(data) > 0:
                # Calculate calibration curve
                fraction_pos, mean_pred = calibration_curve(
                    data['correct'], 
                    data['normalized_confidence'], 
                    n_bins=10,
                    strategy='uniform'
                )
                
                # Plot perfect calibration line
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                
                # Plot actual calibration
                ax.plot(mean_pred, fraction_pos, marker='o', 
                       markersize=8, label=f'{model[:10]}\n{conf_type}')
                
                # Add histogram of predictions
                ax2 = ax.twinx()
                ax2.hist(data['normalized_confidence'], bins=10, alpha=0.3, 
                        color='gray', edgecolor='gray')
                ax2.set_ylabel('Count', color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
                
                # Calculate and display ECE
                ece = calculate_single_ece(data['correct'].values, 
                                          data['normalized_confidence'].values)
                ax.text(0.05, 0.95, f'ECE: {ece:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Mean Predicted Confidence')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'{model} - {conf_type}')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def calculate_single_ece(y_true, y_prob, n_bins=10):
    """Helper function to calculate ECE for a single configuration."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return ece


def analyze_confidence_when_wrong(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze confidence distributions specifically for incorrect predictions.
    This helps understand overconfidence patterns.
    """
    wrong_confidence = []
    
    for (model, conf_type), group in results_df.groupby(['model', 'confidence_type']):
        correct_preds = group[group['correct'] == True]
        incorrect_preds = group[group['correct'] == False]
        
        wrong_confidence.append({
            'Model': model,
            'Confidence Type': conf_type,
            'Mean Conf (Correct)': correct_preds['normalized_confidence'].mean(),
            'Mean Conf (Incorrect)': incorrect_preds['normalized_confidence'].mean(),
            'Median Conf (Correct)': correct_preds['normalized_confidence'].median(),
            'Median Conf (Incorrect)': incorrect_preds['normalized_confidence'].median(),
            'High Conf Errors (>0.8)': (incorrect_preds['normalized_confidence'] > 0.8).mean(),
            'N Correct': len(correct_preds),
            'N Incorrect': len(incorrect_preds)
        })
    
    conf_df = pd.DataFrame(wrong_confidence)
    
    print("\n=== CONFIDENCE ANALYSIS: CORRECT VS INCORRECT PREDICTIONS ===")
    print(conf_df.round(3).to_string(index=False))
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mean confidence comparison
    x = np.arange(len(conf_df))
    width = 0.35
    
    axes[0].bar(x - width/2, conf_df['Mean Conf (Correct)'], width, label='Correct', color='green', alpha=0.7)
    axes[0].bar(x + width/2, conf_df['Mean Conf (Incorrect)'], width, label='Incorrect', color='red', alpha=0.7)
    axes[0].set_xlabel('Model-Confidence Type')
    axes[0].set_ylabel('Mean Confidence')
    axes[0].set_title('Mean Confidence: Correct vs Incorrect Predictions')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{row['Model'][:7]}\n{row['Confidence Type'][:5]}" 
                             for _, row in conf_df.iterrows()], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: High confidence error rate
    axes[1].bar(x, conf_df['High Conf Errors (>0.8)'], color='darkred', alpha=0.7)
    axes[1].set_xlabel('Model-Confidence Type')
    axes[1].set_ylabel('Proportion of Errors with Confidence > 0.8')
    axes[1].set_title('High-Confidence Error Rate')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{row['Model'][:7]}\n{row['Confidence Type'][:5]}" 
                            for _, row in conf_df.iterrows()], rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return conf_df


def create_summary_statistics_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comprehensive summary statistics table for the paper.
    """
    summary_stats = []
    
    for (model, conf_type), group in results_df.groupby(['model', 'confidence_type']):
        # Calculate various metrics
        confidences = group['normalized_confidence'].values
        correct = group['correct'].values
        
        # Consistency across trials
        consistency_scores = []
        for text_id, text_group in group.groupby('text_id'):
            if len(text_group) > 1:
                trial_confs = text_group['normalized_confidence'].values
                cv = np.std(trial_confs) / np.mean(trial_confs) if np.mean(trial_confs) > 0 else 0
                consistency_scores.append(cv)
        
        summary_stats.append({
            'Model': model,
            'Conf Type': conf_type,
            'Accuracy': correct.mean(),
            'Brier Score': np.mean((confidences - correct) ** 2),
            'ECE': calculate_single_ece(correct, confidences),
            'Mean Conf': confidences.mean(),
            'Std Conf': confidences.std(),
            'Consistency (CV)': np.mean(consistency_scores) if consistency_scores else np.nan,
            'Overconf Rate': ((confidences > correct.mean()) & (~correct)).mean()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    print("\n=== COMPREHENSIVE SUMMARY STATISTICS ===")
    print(summary_df.round(3).to_string(index=False))
    
    # Also create a LaTeX table for the paper
    print("\n=== LaTeX Table Format ===")
    print(summary_df.round(3).to_latex(index=False))
    
    return summary_df


# Integration function to call all new analyses
def run_analysis(results_df: pd.DataFrame, dataset_name: str):
    """
    Run all enhanced analyses and generate additional figures/tables.
    """
    print(f"\n{'='*60}")
    print(f"ENHANCED ANALYSIS FOR {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # 1. Detailed calibration table
    calib_table = create_calibration_table(results_df)
    calib_table.to_csv(f'calibration_table_{dataset_name}.csv', index=False)
    
    # 2. ECE and MCE metrics
    ece_mce_df = calculate_ece_mce(results_df)
    ece_mce_df.to_csv(f'ece_mce_metrics_{dataset_name}.csv', index=False)
    
    # 3. Reliability diagrams
    create_reliability_diagram(results_df, f'reliability_diagram_{dataset_name}.png')
    
    # 4. Confidence when wrong analysis
    wrong_conf_df = analyze_confidence_when_wrong(results_df)
    wrong_conf_df.to_csv(f'confidence_when_wrong_{dataset_name}.csv', index=False)
    
    # 5. Summary statistics table
    summary_df = create_summary_statistics_table(results_df)
    summary_df.to_csv(f'summary_statistics_{dataset_name}.csv', index=False)
    
    return {
        'calibration_table': calib_table,
        'ece_mce': ece_mce_df,
        'wrong_confidence': wrong_conf_df,
        'summary_stats': summary_df
    }