#!/bin/bash

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="results_${RUN_TIMESTAMP}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p $WORK_DIR
mkdir -p $LOG_DIR

exec > >(tee -a "${LOG_DIR}/training_run.log") 2>&1

echo "================================================================"
echo "SmartFallMM Fall Detection Training - Run $RUN_TIMESTAMP"
echo "================================================================"

FILTERS=("madgwick" "kalman" "ekf")
SUBJECTS="32,39,30,31,33,34,35,37,43,44,45,36,29,38,46"

cat > "${WORK_DIR}/fix_torch_load.py" << 'EOF'
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
EOF

cat > "${WORK_DIR}/patch_main.py" << 'EOF'
def patch_main():
    import os
    import sys
    main_file = "main.py"
    with open(main_file, "r") as f:
        content = f.read()
    
    # Fix metrics plotting to handle empty arrays
    fixed_content = content.replace(
        "plt.plot(epochs, metrics_history['train_accuracy'], 'b-', label='Training Accuracy')",
        "if 'train_accuracy' in metrics_history and len(metrics_history['train_accuracy']) > 0: plt.plot(epochs[:len(metrics_history['train_accuracy'])], metrics_history['train_accuracy'], 'b-', label='Training Accuracy')"
    )
    
    fixed_content = fixed_content.replace(
        "plt.plot(epochs, metrics_history['val_accuracy'], 'r-', label='Validation Accuracy')",
        "if 'val_accuracy' in metrics_history and len(metrics_history['val_accuracy']) > 0: plt.plot(epochs[:len(metrics_history['val_accuracy'])], metrics_history['val_accuracy'], 'r-', label='Validation Accuracy')"
    )
    
    fixed_content = fixed_content.replace(
        "plt.plot(epochs, metrics_history['train_f1'], 'b-', label='Training F1')",
        "if 'train_f1' in metrics_history and len(metrics_history['train_f1']) > 0: plt.plot(epochs[:len(metrics_history['train_f1'])], metrics_history['train_f1'], 'b-', label='Training F1')"
    )
    
    fixed_content = fixed_content.replace(
        "plt.plot(epochs, metrics_history['val_f1'], 'r-', label='Validation F1')",
        "if 'val_f1' in metrics_history and len(metrics_history['val_f1']) > 0: plt.plot(epochs[:len(metrics_history['val_f1'])], metrics_history['val_f1'], 'r-', label='Validation F1')"
    )
    
    fixed_content = fixed_content.replace(
        "plt.plot(epochs, metrics_history['val_precision'], 'b-', label='Val Precision')",
        "if 'val_precision' in metrics_history and len(metrics_history['val_precision']) > 0: plt.plot(epochs[:len(metrics_history['val_precision'])], metrics_history['val_precision'], 'b-', label='Val Precision')"
    )
    
    fixed_content = fixed_content.replace(
        "plt.plot(epochs, metrics_history['val_recall'], 'r-', label='Val Recall')",
        "if 'val_recall' in metrics_history and len(metrics_history['val_recall']) > 0: plt.plot(epochs[:len(metrics_history['val_recall'])], metrics_history['val_recall'], 'r-', label='Val Recall')"
    )
    
    # Ensure metrics are properly extracted from training results
    fixed_content = fixed_content.replace(
        "train_loss, train_preds, train_labels, train_metrics = train_epoch(",
        "train_loss, train_preds, train_labels, train_metrics = train_epoch("
    )
    
    fixed_content = fixed_content.replace(
        "val_loss, val_preds, val_labels, val_metrics = validate(",
        "val_loss, val_preds, val_labels, val_metrics = validate("
    )
    
    # Fix training metrics collection
    fixed_content = fixed_content.replace(
        "for key, value in {**train_metrics, **val_metrics}.items():",
        "# Ensure train metrics key names are preserved\nfor key, value in train_metrics.items():\n            metrics_history[key].append(value)\n        # Ensure val metrics key names are preserved\n        for key, value in val_metrics.items():"
    )
    
    # Fix path issue for results saving
    fixed_content = fixed_content.replace(
        "csv_path = os.path.join(filter_dir, f'all_folds_results_{args.filter_type}.csv')",
        "csv_path = os.path.join(args.work_dir, f'all_folds_results_{args.filter_type}.csv')"
    )
    
    # Fix results path for overall results
    fixed_content = fixed_content.replace(
        "with open(os.path.join(filter_dir, f'overall_results_{args.filter_type}.json'), 'w') as f:",
        "with open(os.path.join(args.work_dir, f'overall_results_{args.filter_type}.json'), 'w') as f:"
    )
    
    # Update the file
    with open("patched_main.py", "w") as f:
        f.write(fixed_content)
    
    # Notify user
    print("Main.py patched successfully to fix metrics plotting and result saving issues.")
EOF

python "${WORK_DIR}/patch_main.py"

for FILTER in "${FILTERS[@]}"; do
    echo "================================================================"
    echo "Starting training with filter: $FILTER"
    echo "================================================================"
    
    FILTER_DIR="${WORK_DIR}/${FILTER}"
    mkdir -p $FILTER_DIR
    FILTER_LOG="${LOG_DIR}/${FILTER}_training.log"
    echo "Running all folds with filter: $FILTER (logging to $FILTER_LOG)"
    
    PYTHONPATH=$PYTHONPATH:. python -c "
import sys
sys.path.insert(0, '${WORK_DIR}')
from fix_torch_load import *
import importlib.util
spec = importlib.util.spec_from_file_location('main', 'patched_main.py')
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
sys.argv = ['main.py', 
    '--work-dir', '${FILTER_DIR}',
    '--phase', 'train',
    '--fold', '-1',
    '--filter-type', '${FILTER}',
    '--use-gpu', 'True',
    '--device', '0',
    '--seed', '42',
    '--batch-size', '16',
    '--test-batch-size', '32',
    '--num-worker', '4',
    '--fuse', 'True',
    '--model', 'Models.fusion_transformer.FusionTransModel',
    '--optimizer', 'AdamW',
    '--base-lr', '0.0005',
    '--weight-decay', '0.001',
    '--loss', 'bce',
    '--max-epoch', '100',
    '--patience', '20',
    '--use_features', 'False',
    '--subjects', '${SUBJECTS}']
try:
    main.main()
    print('Training completed successfully')
except Exception as e:
    import traceback
    print(f'Error during training: {e}')
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee $FILTER_LOG
    
    if [ $? -eq 0 ]; then
        echo "Completed training with filter: $FILTER successfully"
        
        # Copy the results file to the main directory for easier comparison
        cp "${FILTER_DIR}/overall_results_${FILTER}.json" "${WORK_DIR}/"
        cp "${FILTER_DIR}/all_folds_results_${FILTER}.csv" "${WORK_DIR}/"
    else
        echo "ERROR: Training with filter $FILTER failed with exit code $?"
        echo "Check log file $FILTER_LOG for details"
    fi
    echo "----------------------------------------------------------------"
done

echo "================================================================"
echo "Generating filter comparison"
echo "================================================================"

python -c "
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

work_dir = '${WORK_DIR}'
filters = ['madgwick', 'kalman', 'ekf']

print(f'Comparing results for filters: {filters}')
metrics = {}
all_results = []

for filter_type in filters:
    result_file = os.path.join(work_dir, f'overall_results_{filter_type}.json')
    if os.path.exists(result_file):
        print(f'Loading results from {result_file}')
        with open(result_file, 'r') as f:
            filter_results = json.load(f)
            metrics[filter_type] = filter_results
            
            result_row = {
                'filter': filter_type,
                'num_folds': filter_results.get('num_folds', 0),
                'avg_val_f1': filter_results.get('avg_val_f1', 0),
                'avg_test_f1': filter_results.get('avg_test_f1', 0),
                'avg_test_accuracy': filter_results.get('avg_test_accuracy', 0),
                'avg_test_precision': filter_results.get('avg_test_precision', 0),
                'avg_test_recall': filter_results.get('avg_test_recall', 0)
            }
            all_results.append(result_row)
    else:
        print(f'Warning: No results found for filter {filter_type} at {result_file}')

if metrics:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('avg_test_f1', ascending=False)
    
    csv_path = os.path.join(work_dir, 'filter_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'Saved comparison results to {csv_path}')
    
    plt.figure(figsize=(12, 8))
    
    filter_names = list(metrics.keys())
    x = np.arange(len(filter_names))
    width = 0.2
    
    acc_values = [metrics[f].get('avg_test_accuracy', 0) for f in filter_names]
    f1_values = [metrics[f].get('avg_test_f1', 0) for f in filter_names]
    precision_values = [metrics[f].get('avg_test_precision', 0) for f in filter_names]
    recall_values = [metrics[f].get('avg_test_recall', 0) for f in filter_names]
    
    plt.bar(x - 1.5*width, acc_values, width, label='Accuracy')
    plt.bar(x - 0.5*width, f1_values, width, label='F1')
    plt.bar(x + 0.5*width, precision_values, width, label='Precision')
    plt.bar(x + 1.5*width, recall_values, width, label='Recall')
    
    plt.xlabel('Filter Type')
    plt.ylabel('Average Performance (%)')
    plt.title('Performance Comparison of Different Filters')
    plt.xticks(x, filter_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(acc_values):
        plt.text(i - 1.5*width, v + 1, f'{v:.1f}', ha='center')
    for i, v in enumerate(f1_values):
        plt.text(i - 0.5*width, v + 1, f'{v:.1f}', ha='center')
    for i, v in enumerate(precision_values):
        plt.text(i + 0.5*width, v + 1, f'{v:.1f}', ha='center')
    for i, v in enumerate(recall_values):
        plt.text(i + 1.5*width, v + 1, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'filter_comparison.png'))
    
    plt.figure(figsize=(15, 6))
    plt.axis('off')
    
    table_data = []
    for filter_name in filter_names:
        row = [
            filter_name,
            f\"{metrics[filter_name].get('avg_test_accuracy', 0):.2f}%\",
            f\"{metrics[filter_name].get('avg_test_f1', 0):.2f}%\",
            f\"{metrics[filter_name].get('avg_test_precision', 0):.2f}%\",
            f\"{metrics[filter_name].get('avg_test_recall', 0):.2f}%\"
        ]
        table_data.append(row)
    
    table = plt.table(
        cellText=table_data,
        colLabels=['Filter Type', 'Accuracy', 'F1 Score', 'Precision', 'Recall'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.2, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.suptitle('Filter Comparison Summary', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'filter_comparison_table.png'))
    
    best_filter = results_df.iloc[0]['filter']
    best_f1 = results_df.iloc[0]['avg_test_f1']
    
    comparison = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filters': filter_names,
        'accuracy': {f: metrics[f].get('avg_test_accuracy', 0) for f in filter_names},
        'f1': {f: metrics[f].get('avg_test_f1', 0) for f in filter_names},
        'precision': {f: metrics[f].get('avg_test_precision', 0) for f in filter_names},
        'recall': {f: metrics[f].get('avg_test_recall', 0) for f in filter_names},
        'best_filter': best_filter,
        'best_f1': best_f1
    }
    
    with open(os.path.join(work_dir, 'filter_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print('Filter comparison complete')
    print(f'Best filter: {best_filter} with F1 score: {best_f1:.2f}%')
    
    print('\\nMetrics Summary Table:')
    print('='*80)
    print(f'{"Filter":<10} | {"Accuracy":<10} | {"F1 Score":<10} | {"Precision":<10} | {"Recall":<10}')
    print('-'*80)
    for filter_name in filter_names:
        print(f'{filter_name:<10} | {metrics[filter_name].get(\"avg_test_accuracy\", 0):<10.2f}% | {metrics[filter_name].get(\"avg_test_f1\", 0):<10.2f}% | {metrics[filter_name].get(\"avg_test_precision\", 0):<10.2f}% | {metrics[filter_name].get(\"avg_test_recall\", 0):<10.2f}%')
    print('='*80)
else:
    print('No filter results found for comparison')
"

echo "================================================================"
echo "Saving fold configuration for reference"
echo "================================================================"

python -c "
def create_subject_folds():
    val_subjects = [38, 46]
    always_train_subjects = [45, 36, 29]
    eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
    
    folds = []
    for i, test_subject in enumerate(eligible_subjects):
        test_subjects = [test_subject]
        train_subjects = always_train_subjects + [s for s in eligible_subjects if s != test_subject]
        folds.append({
            'test': test_subjects,
            'val': val_subjects,
            'train': train_subjects
        })
    
    return folds

folds = create_subject_folds()
import json
print('Subject distribution across folds:')
print('-'*80)
for i, fold in enumerate(folds):
    print(f'Fold {i+1}: Test=[{','.join(map(str, fold[\"test\"]))}], Val=[{','.join(map(str, fold[\"val\"]))}], Train={len(fold[\"train\"])} subjects')
print('-'*80)

with open('${WORK_DIR}/fold_configuration.json', 'w') as f:
    json.dump(folds, f, indent=2)
print(f'Fold configuration saved to ${WORK_DIR}/fold_configuration.json')
"

echo "================================================================"
echo "All training complete, results saved to: $WORK_DIR"
echo "================================================================"
echo "See detailed logs in $LOG_DIR directory"
echo "For results comparison, check:"
echo "  $WORK_DIR/filter_comparison.csv"
echo "  $WORK_DIR/filter_comparison.png"
echo "  $WORK_DIR/filter_comparison_table.png"
echo "================================================================"
