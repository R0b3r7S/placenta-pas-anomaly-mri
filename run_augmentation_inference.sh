#!/bin/bash
# ============================================================================
# Same-domain + cross-domain inference for every trained model.
# For each (network, training modality, augmentation cell) model, run
# inference on the clean BTFE test set AND the clean ssh_TSE test set.
#
# Output per model is at:
#   ./runs/<MODEL>/test_on_<TEST_DOMAIN>/{best_model.pth not needed,
#                                         inference_overlays/,
#                                         inference_raw_masks/,
#                                         test_metrics.json,
#                                         slice_metrics.csv}
#
#   networks    : unetplusplus, dynunet
#   aug cells   : none, regular, afa_mixup, afa_cutmix_mixup
#   train mod   : BTFE, SSH_TSE (named "TSE" in run-name for brevity)
#   test mod    : BTFE, SSH_TSE (both → 2 inferences per trained model)
#
#   16 trained models × 2 test sets = 32 inferences
#
# Override env vars to subset:
#   NETWORKS=dynunet  bash run_augmentation_inference.sh
#   CELLS=regular     bash run_augmentation_inference.sh   # one value; edit CELLS=(...) in the script for several
#
# Run after run_augmentation_training.sh:
#   bash run_augmentation_inference.sh
# ============================================================================

# -e: stop on first failure
# -o pipefail: propagate failures through `python | tee` so a python crash
#              doesn't get masked by tee's clean exit.
set -e
set -o pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

BTFE_ROOT="$PROJECT_ROOT/dataset/mri_png/DATASET_BTFE/"
TSE_ROOT="$PROJECT_ROOT/dataset/mri_png/DATASET_SSH_TSE/"

if [ ${#NETWORKS[@]} -eq 0 ]; then
    # DynUNet first (heavier model, results land earlier), then UNet++.
    NETWORKS=("dynunet" "unetplusplus")
fi
if [ ${#TRAIN_MODS[@]} -eq 0 ]; then
    TRAIN_MODS=("BTFE" "TSE")
fi
if [ ${#CELLS[@]} -eq 0 ]; then
    CELLS=("none" "regular" "afa_mixup" "afa_cutmix_mixup")
fi

# --- Timing infrastructure (mirrors run_augmentation_training.sh) ---------
mkdir -p ./runs
TIMING_CSV="./runs/timing_infer.csv"
if [ ! -f "$TIMING_CSV" ]; then
    echo "started_iso,ended_iso,duration_sec,duration_hms,phase,network,modality,cell,test_domain,out_dir,status" > "$TIMING_CSV"
fi

format_hms() {
    local s=$1
    local h=$((s / 3600))
    local m=$(((s / 60) % 60))
    local sec=$((s % 60))
    printf "%02dh%02dm%02ds" "$h" "$m" "$sec"
}

SCRIPT_START=$(date +%s)
SCRIPT_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

on_exit() {
    local exit_code=$?
    local script_end=$(date +%s)
    local script_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local total=$((script_end - SCRIPT_START))
    local total_hms
    total_hms=$(format_hms "$total")
    local status="ok"
    [ "$exit_code" -ne 0 ] && status="interrupted_or_failed"
    echo "$SCRIPT_START_ISO,$script_end_iso,$total,$total_hms,TOTAL_infer,,,,,./runs,$status" >> "$TIMING_CSV"
    echo ""
    echo "================================================================"
    echo "TOTAL inference time: $total_hms ($total s)  status=$status"
    echo "   per-cell timings logged to: $TIMING_CSV"
    echo "================================================================"
}
trap on_exit EXIT

BATCH=8         # test mode is batched now (change H)
WORKERS=8
SEED=42

modality_root() {
    case $1 in
        BTFE) echo "$BTFE_ROOT" ;;
        TSE)  echo "$TSE_ROOT"  ;;
    esac
}

# --- helper ----------------------------------------------------------------
# Args:
#   $1 out_dir        ./runs/<MODEL>/test_on_<DOMAIN>
#   $2 dataset_root   /path/to/test/dataset/
#   $3 checkpoint     /path/to/best_model.pth
#   $4 network        unetplusplus | dynunet
#   $5 train_modality BTFE | TSE      (← used only for the timing CSV)
#   $6 cell           none / regular / afa_mixup / afa_cutmix_mixup
#   $7 test_domain    BTFE | TSE
#   $8+ extra flags   (--use_afa, etc.)
infer() {
    local out_dir=$1
    local dataset_root=$2
    local checkpoint=$3
    local network=$4
    local modality=$5
    local cell=$6
    local test_domain=$7
    shift 7
    mkdir -p "$out_dir"
    local log_file="$out_dir/infer.log"

    local cell_start
    cell_start=$(date +%s)
    local cell_start_iso
    cell_start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "==========================================================="
    echo "INFER  net=$network  out=$out_dir"
    echo "  log: $log_file"
    echo "  tail in another terminal:  tail -f \"$log_file\""
    echo "  started: $cell_start_iso"
    echo "==========================================================="

    local status="ok"
    set +e
    python -u train_placenta_2d_monai_v8.py \
        --dataset_root  "$dataset_root" \
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

    local cell_end
    cell_end=$(date +%s)
    local cell_end_iso
    cell_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local duration=$((cell_end - cell_start))
    local hms
    hms=$(format_hms "$duration")

    echo "$cell_start_iso,$cell_end_iso,$duration,$hms,infer,$network,$modality,$cell,$test_domain,$out_dir,$status" >> "$TIMING_CSV"
    echo "$out_dir: $hms ($duration s)  status=$status" | tee -a "$log_file"

    if [ "$status" = "failed" ]; then
        echo "ERROR: Inference $out_dir failed (exit $exit_code) — see $log_file"
        exit "$exit_code"
    fi
}

# --- evaluate every trained model on every test modality ------------------
# Models trained with AFA (any "afa_*" cell) need --use_afa at test time
# so that the dual-norm layers are reconstructed before loading the weights.
for NETWORK in "${NETWORKS[@]}"; do
    for CELL in "${CELLS[@]}"; do
        # AFA-trained models need --use_afa at inference too.
        EXTRA=""
        if [[ "$CELL" == "afa_"* ]]; then
            EXTRA="--use_afa"
        fi

        for TRAIN in "${TRAIN_MODS[@]}"; do
            MODEL="${TRAIN}_${NETWORK}_${CELL}"
            CKPT="./runs/${MODEL}/fold_0/best_model.pth"

            if [ ! -f "$CKPT" ]; then
                echo "WARNING: Skipping $MODEL — checkpoint not found at $CKPT"
                # Record the skip in the timing CSV so the audit is complete.
                _now_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
                echo "$_now_iso,$_now_iso,0,00h00m00s,infer,$NETWORK,$TRAIN,$CELL,,${CKPT},skipped" >> "$TIMING_CSV"
                continue
            fi

            # Same-domain test
            TRAIN_ROOT=$(modality_root "$TRAIN")
            infer "./runs/${MODEL}/test_on_${TRAIN}" \
                  "$TRAIN_ROOT" \
                  "$CKPT" \
                  "$NETWORK" \
                  "$TRAIN" "$CELL" "$TRAIN" \
                  $EXTRA

            # Cross-domain test
            if [[ "$TRAIN" == "BTFE" ]]; then OTHER="TSE"; else OTHER="BTFE"; fi
            OTHER_ROOT=$(modality_root "$OTHER")
            infer "./runs/${MODEL}/test_on_${OTHER}" \
                  "$OTHER_ROOT" \
                  "$CKPT" \
                  "$NETWORK" \
                  "$TRAIN" "$CELL" "$OTHER" \
                  $EXTRA
        done
    done
done

echo ""
echo "================================================================"
echo "All inference runs complete."
echo "Per-cell outputs are under ./runs/<MODEL>/test_on_<DOMAIN>/"
echo "Aggregate with: python compare_test_metrics.py"
echo "================================================================"
