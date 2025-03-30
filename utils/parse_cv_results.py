# utils/parse_cv_results.py
import json
import sys
import numpy as np
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_cv_results.py <path_to_cv_summary.json> <metric_key1> [metric_key2 ...]")
        sys.exit(1)

    json_path = sys.argv[1]
    metric_keys = sys.argv[2:] # Can request multiple metrics

    if not os.path.exists(json_path):
        print(f"Error: File not found - {json_path}", file=sys.stderr)
        # Output zeros or empty strings for requested metrics to avoid breaking shell script
        print(",".join(["0.00"] * len(metric_keys)))
        sys.exit(0) # Exit gracefully so shell script continues

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file - {json_path}", file=sys.stderr)
        print(",".join(["0.00"] * len(metric_keys)))
        sys.exit(0)
    except Exception as e:
        print(f"Error reading file {json_path}: {e}", file=sys.stderr)
        print(",".join(["0.00"] * len(metric_keys)))
        sys.exit(0)

    fold_metrics = data.get('fold_metrics', [])
    avg_metrics = data.get('average_metrics', {})

    output_values = []

    for key in metric_keys:
        if key.lower() == 'folds': # Special keyword to get fold-by-fold values
             # Print fold metrics for primary keys if available
             primary_metrics = ['accuracy', 'f1', 'precision', 'recall']
             fold_outputs = []
             if fold_metrics:
                 # Try to determine number of folds if possible
                 num_folds_actual = len(fold_metrics)
                 for i in range(num_folds_actual): # Iterate based on actual data
                      fold_data = fold_metrics[i]
                      fold_vals = [
                           f"{fold_data.get('accuracy', 0):.2f}",
                           f"{fold_data.get('f1', fold_data.get('f1_score', 0)):.2f}", # Handle f1_score key
                           f"{fold_data.get('precision', 0):.2f}",
                           f"{fold_data.get('recall', 0):.2f}"
                      ]
                      fold_outputs.append(";".join(fold_vals)) # Use semicolon to separate metrics within a fold
                 output_values.append("|".join(fold_outputs)) # Use pipe to separate folds
             else:
                 output_values.append("NoFoldData") # Placeholder if no fold data

        elif key.lower() == 'avg_all': # Special keyword for average metrics line
             avg_vals = [
                  f"{avg_metrics.get('accuracy', 0):.2f}",
                  f"{avg_metrics.get('f1', 0):.2f}",
                  f"{avg_metrics.get('precision', 0):.2f}",
                  f"{avg_metrics.get('recall', 0):.2f}"
             ]
             output_values.append(";".join(avg_vals))

        else: # Get a specific average metric
            value = avg_metrics.get(key, 0.0)
            output_values.append(f"{value:.2f}") # Format average value

    # Print comma-separated values for shell script parsing
    print(",".join(output_values))

if __name__ == "__main__":
    main()
