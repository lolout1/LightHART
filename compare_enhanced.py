#!/usr/bin/env python
"""
Compare different distillation approaches for fall detection.
"""

import argparse
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare different distillation approaches for fall detection."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/smartfallmm/distill_student_enhanced.yaml",
        help="Base configuration file path"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./comparison_results_enhanced",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0",
        help="GPU device ID"
    )
    return parser.parse_args()

class ComparisonAnalyzer:
    """Compare different distillation approaches."""
    
    def __init__(self, arg):
        """
        Initialize comparison analyzer.
        
        Args:
            arg: Command line arguments
        """
        # Load configuration
        with open(arg.config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set device
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = arg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Approaches to compare
        self.approaches = [
            "standard",      # Standard KD (logits only)
            "enhanced",      # Enhanced KD with multiple sources
            "adversarial"    # Adversarial KD
        ]
        
        # Base directories
        self.base_dirs = {
            "standard": "exps/student_quat",
            "enhanced": "exps/student_enhanced",
            "adversarial": "exps/student_enhanced"  # Same dir but different model
        }
    
    def load_results(self):
        """Load results from all approaches."""
        results = {}
        
        for approach in self.approaches:
            # Look for cross-validation results file
            cv_file = os.path.join(
                self.base_dirs[approach],
                "cross_validation_results.json" if approach == "standard" else "cross_validation_enhanced_results.json"
            )
            
            if os.path.exists(cv_file):
                with open(cv_file, 'r') as f:
                    data = json.load(f)
                    results[approach] = data
            else:
                print(f"Warning: No results found for {approach} at {cv_file}")
        
        return results
    
    def load_training_histories(self):
        """Load training histories from all approaches."""
        histories = {}
        
        for approach in self.approaches:
            # Look for training history files
            history_pattern = os.path.join(
                self.base_dirs[approach],
                "training_history.json" if approach == "standard" else "*enhanced_distillation_history.json"
            )
            
            history_files = glob.glob(history_pattern)
            if history_files:
                with open(history_files[0], 'r') as f:
                    data = json.load(f)
                    histories[approach] = data
            else:
                print(f"Warning: No training history found for {approach} matching {history_pattern}")
        
        return histories
    
    def compare_metrics(self, results):
        """
        Compare metrics across approaches.
        
        Args:
            results: Dictionary with results for each approach
        """
        if not results:
            print("No results to compare.")
            return
        
        # Extract metrics for comparison
        metrics = ["accuracy", "f1", "precision", "recall"]
        
        # Prepare data for plotting
        data = {metric: [] for metric in metrics}
        labels = []
        
        for approach, result in results.items():
            if "average" in result:
                labels.append(approach)
                for metric in metrics:
                    data[metric].append(result["average"].get(metric, 0) * 100)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(labels))
        width = 0.2
        multiplier = 0
        
        for metric, values in data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, values, width, label=metric.capitalize())
            ax.bar_label(rects, fmt='%.1f')
            multiplier += 1
        
        # Add labels and legend
        ax.set_xlabel('Approach')
        ax.set_ylabel('Value (%)')
        ax.set_title('Comparison of Different Distillation Approaches')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels([a.capitalize() for a in labels])
        ax.legend(loc='upper left', ncols=len(metrics))
        
        plt.tight_layout()
        
        # Save figure
        metrics_path = os.path.join(self.output_dir, "approach_comparison.png")
        plt.savefig(metrics_path, dpi=150)
        plt.close()
        
        # Also save as table
        table_path = os.path.join(self.output_dir, "approach_comparison.csv")
        with open(table_path, 'w') as f:
            # Header
            f.write("Approach," + ",".join(metrics) + "\n")
            
            # Data
            for i, approach in enumerate(labels):
                f.write(approach + ",")
                f.write(",".join([f"{data[metric][i]:.2f}" for metric in metrics]))
                f.write("\n")
    
    def compare_learning_curves(self, histories):
        """
        Compare learning curves across approaches.
        
        Args:
            histories: Dictionary with training histories for each approach
        """
        if not histories:
            print("No training histories to compare.")
            return
        
        # Create figure for accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot validation accuracy
        for approach, history in histories.items():
            if "val_acc" in history:
                epochs = range(1, len(history["val_acc"]) + 1)
                ax1.plot(epochs, history["val_acc"], label=f"{approach.capitalize()}")
        
        ax1.set_title('Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot validation loss
        for approach, history in histories.items():
            if "val_loss" in history:
                epochs = range(1, len(history["val_loss"]) + 1)
                ax2.plot(epochs, history["val_loss"], label=f"{approach.capitalize()}")
        
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        curves_path = os.path.join(self.output_dir, "learning_curves_comparison.png")
        plt.savefig(curves_path, dpi=150)
        plt.close()
    
    def compare_fold_performance(self, results):
        """
        Compare fold performance across approaches.
        
        Args:
            results: Dictionary with results for each approach
        """
        if not results:
            print("No results to compare.")
            return
        
        # Create a figure for fold comparison
        plt.figure(figsize=(12, 8))
        
        # Plot F1 scores for each fold
        folds = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
        
        for approach, result in results.items():
            if "folds" in result:
                fold_data = {}
                for fold_info in result["folds"]:
                    fold_name = fold_info.get("fold")
                    if fold_name in folds:
                        fold_data[fold_name] = fold_info.get("f1", 0) * 100
                
                # Plot fold performance
                if fold_data:
                    x = [i for i, fold in enumerate(folds) if fold in fold_data]
                    y = [fold_data[fold] for fold in folds if fold in fold_data]
                    plt.plot(x, y, 'o-', label=approach.capitalize())
        
        plt.xticks(range(len(folds)), folds)
        plt.xlabel('Fold')
        plt.ylabel('F1 Score (%)')
        plt.title('F1 Score by Fold for Different Approaches')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fold_path = os.path.join(self.output_dir, "fold_comparison.png")
        plt.savefig(fold_path, dpi=150)
        plt.close()
    
    def create_report(self):
        """Create comprehensive comparison report."""
        # Load results and histories
        results = self.load_results()
        histories = self.load_training_histories()
        
        # Compare metrics
        self.compare_metrics(results)
        
        # Compare learning curves
        self.compare_learning_curves(histories)
        
        # Compare fold performance
        self.compare_fold_performance(results)
        
        # Create summary report
        report_path = os.path.join(self.output_dir, "comparison_report.md")
        with open(report_path, 'w') as f:
            f.write("# Cross-Modal Distillation Comparison Report\n\n")
            
            f.write("## Summary\n\n")
            if results:
                f.write("| Approach | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |\n")
                f.write("|----------|-------------|--------------|---------------|--------------|\n")
                
                for approach, result in results.items():
                    if "average" in result:
                        avg = result["average"]
                        f.write(f"| {approach.capitalize()} | {avg.get('accuracy', 0)*100:.2f} | {avg.get('f1', 0)*100:.2f} | {avg.get('precision', 0)*100:.2f} | {avg.get('recall', 0)*100:.2f} |\n")
            else:
                f.write("No results available.\n")
            
            f.write("\n## Approach Descriptions\n\n")
            
            f.write("### Standard\n")
            f.write("Standard knowledge distillation using only logits and cross-entropy loss.\n\n")
            
            f.write("### Enhanced\n")
            f.write("Enhanced cross-modal distillation using skeleton and IMU fusion, with multiple knowledge transfer mechanisms:\n")
            f.write("- Logit distillation with KL divergence\n")
            f.write("- Feature alignment between teacher's fused features and student's features\n")
            f.write("- Attention map alignment for better feature attention\n")
            f.write("- Intermediate layer alignment\n")
            f.write("- Contrastive learning for better feature space alignment\n\n")
            
            f.write("### Adversarial\n")
            f.write("Adversarial knowledge distillation with a discriminator network that learns to distinguish between teacher and student features, pushing the student to better mimic the teacher.\n\n")
            f.write("\n## Detailed Analysis\n\n")
            f.write("### Performance by Fold\n")
            if results:
                for approach, result in results.items():
                    if "folds" in result:
                        f.write(f"\n#### {approach.capitalize()}\n\n")
                        f.write("| Fold | Validation Subjects | Best Epoch | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |\n")
                        f.write("|------|---------------------|-----------|--------------|--------------|---------------|--------------|\n")

                        for fold_info in result["folds"]:
                            fold = fold_info.get("fold", "")
                            val_subs = fold_info.get("val_subjects", [])
                            best_epoch = fold_info.get("best_epoch", 0)
                            acc = fold_info.get("accuracy", 0) * 100
                            f1 = fold_info.get("f1", 0) * 100
                            prec = fold_info.get("precision", 0) * 100
                            rec = fold_info.get("recall", 0) * 100

                            f.write(f"| {fold} | {val_subs} | {best_epoch} | {acc:.2f} | {f1:.2f} | {prec:.2f} | {rec:.2f} |\n")
            else:
                f.write("No fold results available.\n")

            f.write("\n### Learning Dynamics\n")
            if histories:
                f.write("See the learning curves comparison figure for detailed training dynamics.\n")

                # Estimate convergence speed
                f.write("\n#### Convergence Analysis\n\n")
                f.write("| Approach | Epochs to 90% Peak | Epochs to 95% Peak | Peak Validation Acc (%) | Peak Validation F1 (%) |\n")
                f.write("|----------|---------------------|-------------------|-------------------------|------------------------|\n")

                for approach, history in histories.items():
                    if "val_acc" in history and "val_f1" in history:
                        val_acc = history["val_acc"]
                        val_f1 = history["val_f1"]

                        peak_acc = max(val_acc)
                        peak_f1 = max(val_f1)

                        # Find epochs to reach percentage of peak
                        acc_90_epoch = next((i+1 for i, acc in enumerate(val_acc) if acc >= 0.9 * peak_acc), "-")
                        acc_95_epoch = next((i+1 for i, acc in enumerate(val_acc) if acc >= 0.95 * peak_acc), "-")

                        f.write(f"| {approach.capitalize()} | {acc_90_epoch} | {acc_95_epoch} | {peak_acc:.2f} | {peak_f1:.2f} |\n")
            else:
                f.write("No learning dynamics data available.\n")

            f.write("\n## Conclusion\n\n")
            if results:
                best_approach = max(results.items(), key=lambda x: x[1]["average"].get("f1", 0))[0]
                f.write(f"Based on the comprehensive analysis, the **{best_approach.capitalize()}** approach provides the best performance for cross-modal distillation in fall detection.\n\n")

                # Calculate improvement percentages
                if "standard" in results and best_approach != "standard":
                    std_f1 = results["standard"]["average"].get("f1", 0)
                    best_f1 = results[best_approach]["average"].get("f1", 0)
                    improvement = ((best_f1 / std_f1) - 1) * 100 if std_f1 > 0 else 0

                    f.write(f"The {best_approach} approach improves F1 score by {improvement:.1f}% compared to standard knowledge distillation.\n\n")

                f.write("### Key Takeaways\n\n")
                f.write("1. **Cross-Modal Knowledge Transfer**: Successfully transferring skeleton knowledge to IMU-only model\n")
                f.write("2. **Feature Alignment**: Aligning feature spaces between teacher and student is crucial\n")
                f.write("3. **Attention Mechanism**: Attention map alignment helps student focus on important data aspects\n")
                f.write("4. **Rich Knowledge Sources**: Using multiple knowledge sources (logits, features, attention) improves performance\n")
                f.write("5. **Orientation Tracking**: Quaternion-based orientation tracking provides a strong foundation for fall detection\n")
            else:
                f.write("No conclusion could be drawn due to lack of result data.\n")

        print(f"Comprehensive report created at {report_path}")

def main():
    """Main function."""
    # Parse arguments
    arg = parse_args()

    # Run comparison
    analyzer = ComparisonAnalyzer(arg)
    analyzer.create_report()

    print(f"Comparison analysis completed. Results saved to {arg.output_dir}")

if __name__ == "__main__":
    main()
