#!/bin/bash
# ============================================================================
# External-validation inference: every trained model evaluated on the
# DATASET_EXTERNAL_REBRO held-out cohort.
#
# Output per model:
#   ./runs/<MODEL>/test_on_EXTERNAL_REBRO/{inference_overlays/,
#                                          inference_raw_masks/,
#                                          test_metrics.json,
#                                          slice_metrics.csv,
#                                          per_patient_metrics.json}
#
#   networks   : unetplusplus, dynunet
#   aug cells  : none, regular, afa_mixup, afa_cutmix_mixup
#   train mod  : BTFE, SSH_TSE (named "TSE" in run-name)
#
#   16 trained models × 1 external test set = 16 inferences
#
# Prereqs:
#   - All 16 models trained via run_augmentation_training.sh
#   - DATASET_EXTERNAL_REBRO built via build_external_test_set.py
#
# Override env vars to subset:
#   NETWORKS=dynunet  bash run_external_inference.sh
#   CELLS=regular     bash run_external_inference.sh   # one value; edit CELLS=(...) in the script for several
# ============================================================================

set -e
set -o pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

EXTERNAL_ROOT="$PROJECT_ROOT/dataset/mri_png/DATASET_EXTERNAL_REBRO/"

if [ ! -d "$EXTERNAL_ROOT" ]; then
    echo "ERROR: $EXTERNAL_ROOT not found — run build_external_test_set.py first."
    exit 1
fi

if [ ${#NETWORKS[@]} -eq 0 ]; then
    NETWORKS=("dynunet" "unetplusplus")
fi
if [ ${#TRAIN_MODS[@]} -eq 0 ]; then
    TRAIN_MODS=("BTFE" "TSE")
fi
if [ ${#CELLS[@]} -eq 0 ]; then
    CELLS=("none" "regular" "afa_mixup" "afa_cutmix_mixup")
fi

# --- Timing infrastructure -------------------------------------------------
mkdir -p ./runs
TIMING_CSV="./runs/timing_external.csv"
if [ ! -f "$TIMING_CSV" ]; then
    echo "started_iso,ended_iso,duration_sec,duration_hms,phase,network,modality,cell,test_domain,out_dir,status" > "$TIMING_CSV"
fi

format_hms() {
    local s=$1
    printf "%02dh%02dm%02ds" $((s/3600)) $(((s/60)%60)) $((s%60))
}

SCRIPT_START=$(date +%s)
SCRIPT_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

on_exit() {
    local exit_code=$?
    local script_end script_end_iso total total_hms status
    script_end=$(date +%s)
    script_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    total=$((script_end - SCRIPT_START))
    total_hms=$(format_hms "$total")
    status="ok"
    [ "$exit_code" -ne 0 ] && status="interrupted_or_failed"
    echo "$SCRIPT_START_ISO,$script_end_iso,$total,$total_hms,TOTAL_external,,,,,./runs,$status" >> "$TIMING_CSV"
    echo ""
    echo "================================================================"
    echo "TOTAL external-validation time: $total_hms ($total s)  status=$status"
    echo "   per-cell timings logged to: $TIMING_CSV"
    echo "================================================================"
}
trap on_exit EXIT

BATCH=8
WORKERS=8
SEED=42

infer() {
    local out_dir=$1
    local checkpoint=$2
    local network=$3
    local modality=$4
    local cell=$5
    shift 5
    mkdir -p "$out_dir"
    local log_file="$out_dir/infer.log"

    local cell_start cell_start_iso
    cell_start=$(date +%s)
    cell_start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "==========================================================="
    echo "EXTERNAL  net=$network  train=$modality  cell=$cell"
    echo "  out: $out_dir"
    echo "  log: $log_file"
    echo "  tail in another terminal:  tail -f \"$log_file\""
    echo "==========================================================="

    local status="ok"
    set +e
    python -u train_placenta_2d_monai_v8.py \
        --dataset_root  "$EXTERNAL_ROOT" \
        --out_dir       "$out_dir" \
        --checkpoint    "$checkpoint" \
        --mode          test \
        --network       "$network" \
        --cv_folds      1 \
        --batch_size    "$BATCH" \
        --num_workers   "$WORKERS" \
        --amp \
        --seed          "$SEED" \
        "$@" \
        2>&1 | tee -a "$log_file"
    local exit_code=$?
    set -e

    [ "$exit_code" -ne 0 ] && status="failed"

    local cell_end cell_end_iso duration hms
    cell_end=$(date +%s)
    cell_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    duration=$((cell_end - cell_start))
    hms=$(format_hms "$duration")

    echo "$cell_start_iso,$cell_end_iso,$duration,$hms,external,$network,$modality,$cell,EXTERNAL_REBRO,$out_dir,$status" >> "$TIMING_CSV"
    echo "$out_dir: $hms ($duration s)  status=$status" | tee -a "$log_file"

    if [ "$status" = "failed" ]; then
        echo "ERROR: Inference $out_dir failed (exit $exit_code) — see $log_file"
        exit "$exit_code"
    fi
}

# --- run every trained model on the external set --------------------------
for NETWORK in "${NETWORKS[@]}"; do
    for CELL in "${CELLS[@]}"; do
        EXTRA=""
        if [[ "$CELL" == "afa_"* ]]; then
            EXTRA="--use_afa"
        fi

        for TRAIN in "${TRAIN_MODS[@]}"; do
            MODEL="${TRAIN}_${NETWORK}_${CELL}"
            CKPT="./runs/${MODEL}/fold_0/best_model.pth"

            if [ ! -f "$CKPT" ]; then
                echo "WARNING: Skipping $MODEL — checkpoint not found at $CKPT"
                _now_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
                echo "$_now_iso,$_now_iso,0,00h00m00s,external,$NETWORK,$TRAIN,$CELL,EXTERNAL_REBRO,${CKPT},skipped" >> "$TIMING_CSV"
                continue
            fi

            OUT_DIR="./runs/${MODEL}/test_on_EXTERNAL_REBRO"
            # Resume support: if test_metrics.json already exists for this cell,
            # skip it. Set FORCE=1 to redo everything from scratch.
            if [ -f "$OUT_DIR/fold_0/test_metrics.json" ] && [ "${FORCE:-0}" != "1" ]; then
                echo "Skipping $MODEL — already complete (set FORCE=1 to redo)"
                _now_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
                echo "$_now_iso,$_now_iso,0,00h00m00s,external,$NETWORK,$TRAIN,$CELL,EXTERNAL_REBRO,$OUT_DIR,already_done" >> "$TIMING_CSV"
                continue
            fi

            infer "$OUT_DIR" \
                  "$CKPT" \
                  "$NETWORK" \
                  "$TRAIN" "$CELL" \
                  $EXTRA
        done
    done
done

echo ""
echo "================================================================"
echo "External-validation inference complete."
echo "Per-model outputs are under ./runs/<MODEL>/test_on_EXTERNAL_REBRO/"
echo "Aggregate with: python compare_test_metrics.py"
echo "================================================================"
