#!/usr/bin/env python3
import os, json, argparse, numpy as np, glob
from typing import List, Dict, Any

def load_fold_results(output_dir: str) -> List[Dict[str, Any]]:
    fold_metrics = []
    fold_dirs = sorted(glob.glob(os.path.join(output_dir, "fold_*")))
    for i, fold_dir in enumerate(fold_dirs, 1):
        results_file = os.path.join(fold_dir, "fold_summary.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f: results = json.load(f)
                fold_metrics.append({
                    'fold': i,
                    'accuracy': results.get('test_metrics', {}).get('accuracy', 0),
                    'f1': results.get('test_metrics', {}).get('f1', 0),
                    'precision': results.get('test_metrics', {}).get('precision', 0),
                    'recall': results.get('test_metrics', {}).get('recall', 0),
                    'balanced_accuracy': results.get('test_metrics', {}).get('balanced_accuracy', 0),
                    'train_subjects': results.get('train_subjects', []),
                    'test_subjects': results.get('test_subjects', [])
                })
                print(f"Loaded results from {results_file}")
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
    return fold_metrics

def create_cv_summary(fold_metrics: List[Dict[str, Any]], filter_type: str) -> Dict[str, Any]:
    if not fold_metrics:
        return {"filter_type": filter_type, "average_metrics": {}, "fold_metrics": [], "test_configs": []}
    
    metrics = ["accuracy", "f1", "precision", "recall", "balanced_accuracy"]
    avg_metrics = {}
    
    for metric in metrics:
        values = [fold.get(metric, 0) for fold in fold_metrics]
        if values:
            avg_metrics[metric] = float(np.mean(values))
            avg_metrics[f"{metric}_std"] = float(np.std(values))
        else:
            avg_metrics[metric] = 0
            avg_metrics[f"{metric}_std"] = 0
    
    test_configs = []
    for fold in fold_metrics:
        test_configs.append({
            'fold_id': fold.get('fold', 0),
            'train_subjects': fold.get('train_subjects', []),
            'test_subjects': fold.get('test_subjects', []),
            'metrics': {
                'accuracy': fold.get('accuracy', 0),
                'f1': fold.get('f1', 0),
                'precision': fold.get('precision', 0),
                'recall': fold.get('recall', 0),
                'balanced_accuracy': fold.get('balanced_accuracy', 0),
            }
        })
    
    return {
        "filter_type": filter_type, 
        "average_metrics": avg_metrics, 
        "fold_metrics": fold_metrics,
        "test_configs": test_configs
    }

def main():
    parser = argparse.ArgumentParser(description="Recover CV summary from fold results")
    parser.add_argument("--output-dir", required=True, help="Model output directory")
    parser.add_argument("--filter-type", required=True, help="Filter type (madgwick, kalman, ekf, none)")
    args = parser.parse_args()
    
    fold_metrics = load_fold_results(args.output_dir)
    cv_summary = create_cv_summary(fold_metrics, args.filter_type)
    
    summary_path = os.path.join(args.output_dir, "test_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
        
    print(f"Recovered CV summary saved to {summary_path}")

if __name__ == "__main__":
    main()
