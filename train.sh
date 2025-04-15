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
VAL_SUBJECTS="38,46"
FIXED_TRAIN_SUBJECTS="45,36,29"
ELIGIBLE_TEST_SUBJECTS="32,39,30,31,33,34,35,37,43,44"
ALL_SUBJECTS="${ELIGIBLE_TEST_SUBJECTS},${FIXED_TRAIN_SUBJECTS},${VAL_SUBJECTS}"

# Fix torch compatibility for older torch versions
python -c "
import torch
if hasattr(torch, '__version__'):
    ver = torch.__version__.split('.')
    if int(ver[0]) >= 1 and int(ver[1]) >= 13:
        print('Applying torch.load patch for compatibility')
        from functools import partial
        original_load = torch.load
        torch.load = partial(original_load, weights_only=False)
"

# Process each filter type
for FILTER in "${FILTERS[@]}"; do
    echo "================================================================"
    echo "Starting training with filter: $FILTER"
    echo "================================================================"
    
    FILTER_DIR="${WORK_DIR}/${FILTER}"
    mkdir -p $FILTER_DIR
    FILTER_LOG="${LOG_DIR}/${FILTER}_training.log"
    echo "Running all folds with filter: $FILTER (logging to $FILTER_LOG)"
    
    # Run training
    python main.py \
        --work-dir $FILTER_DIR \
        --phase train \
        --fold -1 \
        --filter-type $FILTER \
        --use-gpu True \
        --device 0 \
        --seed 42 \
        --batch-size 16 \
        --test-batch-size 32 \
        --num-worker 4 \
        --fuse True \
        --model Models.fusion_transformer.FusionTransModel \
        --optimizer adamw \
        --base-lr 0.0005 \
        --weight-decay 0.001 \
        --loss bce \
        --max-epoch 100 \
        --patience 20 \
        --use_features False \
        --subjects $ALL_SUBJECTS 2>&1 | tee $FILTER_LOG
    
    if [ $? -eq 0 ]; then
        echo "Completed training with filter: $FILTER successfully"
        
        # Copy results to main directory
        if [ -f "${FILTER_DIR}/overall_results_${FILTER}.json" ]; then
            cp "${FILTER_DIR}/overall_results_${FILTER}.json" "${WORK_DIR}/"
        else
            echo "Warning: overall_results_${FILTER}.json not found"
        fi
        
        if [ -f "${FILTER_DIR}/all_folds_results_${FILTER}.csv" ]; then
            cp "${FILTER_DIR}/all_folds_results_${FILTER}.csv" "${WORK_DIR}/"
        else
            echo "Warning: all_folds_results_${FILTER}.csv not found"
        fi
    else
        echo "ERROR: Training with filter $FILTER failed"
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
        try:
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
        except Exception as e:
            print(f'Error loading results for {filter_type}: {e}')
    else:
        print(f'Warning: No results found for filter {filter_type} at {result_file}')

if metrics:
    # Create CSV comparison
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('avg_test_f1', ascending=False)
    csv_path = os.path.join(work_dir, 'filter_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'Saved comparison results to {csv_path}')
    
    try:
        # Create bar chart
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
        plt.savefig(os.path.join(work_dir, 'filter_comparison.png'), dpi=300)
        
        # Create table visualization
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
        plt.savefig(os.path.join(work_dir, 'filter_comparison_table.png'), dpi=300)
        
        # Save comparison data to JSON
        best_filter = results_df.iloc[0]['filter'] if not results_df.empty else 'none'
        best_f1 = results_df.iloc[0]['avg_test_f1'] if not results_df.empty else 0
        
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
        print(f\"{'Filter':<10} | {'Accuracy':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10}\")
        print('-'*80)
        for filter_name in filter_names:
            print(f\"{filter_name:<10} | {metrics[filter_name].get('avg_test_accuracy', 0):<10.2f}% | {metrics[filter_name].get('avg_test_f1', 0):<10.2f}% | {metrics[filter_name].get('avg_test_precision', 0):<10.2f}% | {metrics[filter_name].get('avg_test_recall', 0):<10.2f}%\")
        print('='*80)
    except Exception as e:
        print(f'Error creating visualizations: {e}')
        import traceback
        traceback.print_exc()
else:
    print('No filter results found for comparison')
"

echo "================================================================"
echo "Analyzing cross-validation split for data leakage verification"
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
import numpy as np

print('Cross-validation fold verification:')
print('-'*80)

# Verify no overlap between test sets
all_test_subjects = []
for i, fold in enumerate(folds):
    all_test_subjects.extend(fold['test'])
    print(f\"Fold {i+1}: Test=[{','.join(map(str, fold['test']))}], Val=[{','.join(map(str, fold['val']))}], Train={len(fold['train'])} subjects\")
    
    # Verify no overlap between train/val/test
    train_set = set(fold['train'])
    val_set = set(fold['val'])
    test_set = set(fold['test'])
    
    if train_set & val_set:
        print(f\"ERROR: Overlap between train and validation sets in fold {i+1}: {train_set & val_set}\")
    if train_set & test_set:
        print(f\"ERROR: Overlap between train and test sets in fold {i+1}: {train_set & test_set}\")
    if val_set & test_set:
        print(f\"ERROR: Overlap between validation and test sets in fold {i+1}: {val_set & test_set}\")

# Verify each eligible test subject appears exactly once in test
unique_test_subjects = set(all_test_subjects)
eligible_test_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]

print('-'*80)
print(f'Unique test subjects across all folds: {sorted(unique_test_subjects)}')
print(f'Eligible test subjects: {sorted(eligible_test_subjects)}')

if set(eligible_test_subjects) != unique_test_subjects:
    missing = set(eligible_test_subjects) - unique_test_subjects
    extra = unique_test_subjects - set(eligible_test_subjects)
    if missing:
        print(f\"ERROR: Some eligible test subjects never appear in test set: {missing}\")
    if extra:
        print(f\"ERROR: Some subjects in test sets are not eligible: {extra}\")
else:
    print('✓ All eligible test subjects appear in test sets')

# Check for duplicates in test sets
subject_counts = {}
for subj in all_test_subjects:
    subject_counts[subj] = subject_counts.get(subj, 0) + 1

duplicates = {subj: count for subj, count in subject_counts.items() if count > 1}
if duplicates:
    print(f\"ERROR: Some subjects appear in multiple test sets: {duplicates}\")
else:
    print('✓ No subject appears in multiple test sets')

# Verify fixed train and validation subjects
fixed_train_subjects = [45, 36, 29]
fixed_val_subjects = [38, 46]

for i, fold in enumerate(folds):
    missing_train = set(fixed_train_subjects) - set(fold['train'])
    if missing_train:
        print(f\"ERROR: Fixed training subjects missing in fold {i+1}: {missing_train}\")
    
    missing_val = set(fixed_val_subjects) - set(fold['val'])
    if missing_val:
        print(f\"ERROR: Fixed validation subjects missing in fold {i+1}: {missing_val}\")

if all(set(fixed_train_subjects).issubset(set(fold['train'])) for fold in folds):
    print('✓ All fixed training subjects appear in every train set')
    
if all(set(fixed_val_subjects) == set(fold['val']) for fold in folds):
    print('✓ All fixed validation subjects appear in every validation set')

print('-'*80)
try:
    with open('${WORK_DIR}/fold_verification.json', 'w') as f:
        json.dump({
            'folds': folds,
            'verification': {
                'unique_test_subjects': list(unique_test_subjects),
                'eligible_test_subjects': eligible_test_subjects,
                'fixed_train_subjects': fixed_train_subjects,
                'fixed_val_subjects': fixed_val_subjects,
                'all_passing': (set(eligible_test_subjects) == unique_test_subjects and 
                               not duplicates and
                               all(set(fixed_train_subjects).issubset(set(fold['train'])) for fold in folds) and
                               all(set(fixed_val_subjects) == set(fold['val']) for fold in folds))
            }
        }, f, indent=2)
    print(f'Fold verification saved to ${WORK_DIR}/fold_verification.json')
except Exception as e:
    print(f'Error saving fold verification: {e}')
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
