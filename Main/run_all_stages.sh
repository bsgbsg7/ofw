#!/bin/bash
#
# 依次运行 train_stage1 → train_stage2 → train_stage3
# 用法:
#   ./run_all_stages.sh            # 前台运行
#   ./run_all_stages.sh --tmux     # 在 tmux 会话中运行（推荐）
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="ofw"
TMUX_SESSION="train_ofw"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_stage() {
    local stage=$1
    log "========== Starting ${stage} =========="
    python "${SCRIPT_DIR}/${stage}.py"
    log "========== ${stage} completed =========="
}

train_all() {
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"

    log "Python: $(which python)"
    log "TF visible GPUs: $(python -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))')"

    run_stage "train_stage1"
    run_stage "train_stage2"
    run_stage "train_stage3"

    log "All stages completed."
}

if [[ "${1:-}" == "--tmux" ]]; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        echo "tmux session '${TMUX_SESSION}' already exists. Attach with: tmux attach -t ${TMUX_SESSION}"
        exit 1
    fi
    tmux new-session -d -s "${TMUX_SESSION}" \
        "cd ${SCRIPT_DIR} && bash '$0' 2>&1 | tee ${SCRIPT_DIR}/run_all.log"
    echo "Training started in tmux session '${TMUX_SESSION}'"
    echo "  Attach:  tmux attach -t ${TMUX_SESSION}"
    echo "  Log:     tail -f ${SCRIPT_DIR}/run_all.log"
else
    train_all
fi
