#!/bin/bash
# Script to compare Madgwick, Kalman, and Acc+Gyro only models fold-by-fold
# Uses STATEFUL filtering & LINEAR acceleration input.
# Avoids jq dependency by using a Python helper script.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
DEVICE="<span class="math-inline">\{CUDA\_VISIBLE\_DEVICES\:\-0\}"
BASE\_LR\=0\.0005
WEIGHT\_DECAY\=0\.001
NUM\_EPOCHS\=60
PATIENCE\=15
SEED\=42
BATCH\_SIZE\=16
PARALLEL\_THREADS\=24
\# Directories
BASE\_DIR\=</span>(pwd)
RESULTS_DIR="<span class="math-inline">\{BASE\_DIR\}/filter\_comparison\_run\_</span>(date +%Y%m%d_%H%M%S)"
CONFIG_DIR="<span class="math-inline">\{BASE\_DIR\}/config/comparison"
LOG\_FILE\="</span>{RESULTS_DIR}/comparison_run.log"
PYTHON_EXE="python" # Use python or python3 as appropriate for your env
PARSER_SCRIPT="<span class="math-inline">\{BASE\_DIR\}/utils/parse\_cv\_results\.py"
\# Models/Configs to Compare
declare \-A MODELS\=\(
\["Madgwick\_Stateful"\]\="madgwick\_stateful\.yaml"
\["Kalman\_Stateful"\]\="kalman\_stateful\.yaml"
\["AccGyro\_Only"\]\="acc\_gyro\_only\.yaml"
\)
MODEL\_KEYS\=\("</span>{!MODELS[@]}") # Get keys in declared order

# --- Helper Functions ---
timestamp() { date +"%Y-%m-%d_%H:%M:%S"; }
log() { echo "[$(timestamp)] [$1] <span class="math-inline">2" \| tee \-a "</span>{LOG_FILE}"; }
check_status() {
    local status=$?; local message=$1
    if [ <span class="math-inline">status \-ne 0 \]; then log "ERROR" "</span>{message} failed with status ${status}"; exit <span class="math-inline">status;
else log "INFO" "</span>{message} completed successfully."; fi; return <span class="math-inline">status
\}
\# \-\-\- Main Execution \-\-\-
mkdir \-p "</span>{RESULTS_DIR}"
> "${LOG_FILE}" # Clear log file

log "INFO" "========================================================="
log "INFO" " Starting Filter Comparison Script (jq-free)"
log "INFO" "========================================================="
log "INFO" "Results Directory: ${RESULTS_DIR}"
log "INFO" "Using GPU(s): ${DEVICE}"

# Check for Python and Parser Script
if ! command -v <span class="math-inline">\{PYTHON\_EXE\} &\> /dev/null; then log "ERROR" "</span>{PYTHON_EXE} not found in PATH."; exit 1; fi
if [ ! -f "${PARSER_SCRIPT}" ]; then log "ERROR" "Parser script not found: <span class="math-inline">\{PARSER\_SCRIPT\}"; exit 1; fi
log "INFO" "</span>{PYTHON_EXE} and parser script found."

# Ensure config files exist (create if necessary - assumes they are present)
mkdir -p "<span class="math-inline">\{CONFIG\_DIR\}"
for config\_fname in "</span>{MODELS[@]}"; do
    if [ ! -f "<span class="math-inline">\{CONFIG\_DIR\}/</span>{config_fname}" ]; then
        log "ERROR" "Config file <span class="math-inline">\{CONFIG\_DIR\}/</span>{config_fname} missing. Please create it (e.g., copy from previous response)."
        exit 1
    fi
done
log "INFO" "Configuration files found."

# Train models sequentially
for model_key in "<span class="math-inline">\{MODEL\_KEYS\[@\]\}"; do
config\_fname\="</span>{MODELS[<span class="math-inline">model\_key\]\}"
config\_file\="</span>{CONFIG_DIR}/<span class="math-inline">\{config\_fname\}"
output\_dir\="</span>{RESULTS_DIR}/${model_key}"

    log "INFO" ">>> Training Model: ${model_key} <<<"
    log "INFO" "Config: ${config_file}"
    log "INFO" "Output: <span class="math-inline">\{output\_dir\}"
mkdir \-p "</span>{output_dir}/logs"

    CUDA_VISIBLE_DEVICES=${DEVICE} <span class="math-inline">\{PYTHON\_EXE\} main\.py \\
\-\-config "</span>{config_file}" \
        --work-dir "<span class="math-inline">\{output\_dir\}" \\
\-\-model\-saved\-name "</span>{model_key}_best" \
        --device 0 \
        --multi-gpu False \
        --kfold True \
        --parallel-threads ${PARALLEL_THREADS} \
        --num-epoch ${NUM_EPOCHS} \
        --patience ${PATIENCE} \
        --seed <span class="math-inline">\{SEED\} 2\>&1 \| tee "</span>{output_dir}/logs/training.log"
    check_status "Training <span class="math-inline">\{model\_key\}"
\[ \! \-f "</span>{output_dir}/cv_summary.json" ] && log "WARNING" "cv_summary.json not found for <span class="math-inline">\{model\_key\}\."
done
log "INFO" "All training runs completed\."
\# Compare results using Python parser
log "INFO" "\-\-\- Generating Fold\-by\-Fold Comparison \-\-\-"
declare \-A FOLD\_RESULTS \# Associative array to store fold data \[model\_key, fold\_idx\] \-\> "Acc;F1;Prec;Rec"
declare \-A AVG\_RESULTS \# Associative array for averages \[model\_key\] \-\> "Acc;F1;Prec;Rec"
num\_folds\=0
\# Fetch data using the Python parser
for model\_key in "</span>{MODEL_KEYS[@]}"; do
    summary_file="<span class="math-inline">\{RESULTS\_DIR\}/</span>{model_key}/cv_summary.json"
    if [ -f "<span class="math-inline">\{summary\_file\}" \]; then
\# Get fold data string \(metrics separated by ';', folds separated by '\|'\)
fold\_data\_str\=</span>(<span class="math-inline">\{PYTHON\_EXE\} "</span>{PARSER_SCRIPT}" "${summary_file}" "folds")
        check_status "Parsing fold data for <span class="math-inline">\{model\_key\}"
\# Get average data string \(metrics separated by ';'\)
avg\_data\_str\=</span>(<span class="math-inline">\{PYTHON\_EXE\} "</span>{PARSER_SCRIPT}" "${summary_file}" "avg_all")
        check_status "Parsing average data for <span class="math-inline">\{model\_key\}"
AVG\_RESULTS\["</span>{model_key
