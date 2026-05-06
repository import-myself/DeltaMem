#!/usr/bin/env bash
# =============================================================================
# PRTree 消融实验统一入口（调度器）
#
# 用法：bash run_ablation.sh [实验类型]
#
#   threshold      → 阈值参数消融（ALFWorld only）
#                    → 调用 run_threshold_ablation.sh
#
#   memory_mode    → 记忆模式消融（3 benchmarks）
#                    → 调用 run_memory_mode_ablation.sh
#                    也可单独跑某个 benchmark：
#                    → run_memory_mode_alfworld.sh
#                    → run_memory_mode_sciworld.sh
#                    → run_memory_mode_mind2web.sh
#
#   all            → 顺序运行以上两组（默认）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="${1:-all}"

case "${MODE}" in
    threshold)
        bash run_threshold_ablation.sh
        ;;
    memory_mode)
        bash run_memory_mode_ablation.sh
        ;;
    all)
        bash run_threshold_ablation.sh
        bash run_memory_mode_ablation.sh
        ;;
    *)
        echo "Usage: bash run_ablation.sh [threshold|memory_mode|all]"
        exit 1
        ;;
esac
