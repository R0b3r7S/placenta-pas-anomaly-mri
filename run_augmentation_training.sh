#!/bin/bash
# ============================================================================
# Train the segmentation benchmark.
#
#   networks    : unetplusplus, dynunet
#   aug cells   : none, regular, afa_mixup, afa_cutmix_mixup
#   modalities  : BTFE, SSH_TSE
#
#   2 networks × 4 cells × 2 modalities = 16 training runs.
#
# All runs use a single fold (--cv_folds 1); the train/val/test split is read
# from <dataset_root>/splits.json.
#
# To run only one network, override NETWORKS at call time:
#   NETWORKS=unetplusplus  bash run_augmentation_training.sh
#   NETWORKS=dynunet       bash run_augmentation_training.sh   # the networks reported in the paper
#
# To run only one modality, override MODALITIES:
#   MODALITIES=BTFE        bash run_augmentation_training.sh
#
# Default = train everything:
#   bash run_augmentation_training.sh
# ============================================================================

# -e: stop on first failure
# -o pipefail: propagate failures through `python | tee` so a python crash
#              doesn't get masked by tee's clean exit.
set -e
set -o pipefail

# --- paths (relative to the repo root) -------------------------------------
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

BTFE_ROOT="$PROJECT_ROOT/dataset/mri_png/DATASET_BTFE/"
TSE_ROOT="$PROJECT_ROOT/dataset/mri_png/DATASET_SSH_TSE/"

# --- which networks and modalities to run ---------------------------------
# Defaults: train all 4 combos. Override at call time with e.g.
#   NETWORKS=unetplusplus MODALITIES=BTFE bash run_augmentation_training.sh
#
# Order matters: DynUNet runs FIRST (heavier model, results land earlier),
# then UNet++. You can flip the order by setting NETWORKS yourself.
if [ ${#NETWORKS[@]} -eq 0 ]; then
    NETWORKS=("dynunet" "unetplusplus")
fi
if [ ${#MODALITIES[@]} -eq 0 ]; then
    MODALITIES=("BTFE" "TSE")
fi

# --- training hyper-parameters --------------------------------------------
EPOCHS=150
BATCH=8
LR=0.001
WEIGHT_DECAY=1e-5
PATIENCE=25
WORKERS=8
SEED=42

# RESUME=1 → pass --resume to the trainer so it picks up from
# <out_dir>/last_checkpoint.pth if the previous run was interrupted.
# Default 0 = train from scratch each invocation. Toggle at call time:
#   RESUME=1 bash run_augmentation_training.sh
RESUME=${RESUME:-0}
RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume"
fi

# --- Timing infrastructure -------------------------------------------------
# Every cell's wall-clock is appended to runs/timing_train.csv with status
# ok / failed. A TOTAL row is added at the end. Tail this file in another
# terminal to track progress:
#   tail -f runs/timing_train.csv
mkdir -p ./runs
TIMING_CSV="./runs/timing_train.csv"
if [ ! -f "$TIMING_CSV" ]; then
    echo "started_iso,ended_iso,duration_sec,duration_hms,phase,network,modality,cell,out_dir,status" > "$TIMING_CSV"
fi

format_hms() {
    local s=$1
    local h=$((s / 3600))
    local m=$(((s / 60) % 60))
    local sec=$((s % 60))
    printf "%02dh%02dm%02ds" "$h" "$m" "$sec"
}

# Total time across the whole script
SCRIPT_START=$(date +%s)
SCRIPT_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Cleanup trap: even if the script is Ctrl-C'd or fails, write the TOTAL row.
on_exit() {
    local exit_code=$?
    local script_end=$(date +%s)
    local script_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local total=$((script_end - SCRIPT_START))
    local total_hms
    total_hms=$(format_hms "$total")
    local status="ok"
    [ "$exit_code" -ne 0 ] && status="interrupted_or_failed"
    echo "$SCRIPT_START_ISO,$script_end_iso,$total,$total_hms,TOTAL_train,,,,./runs,$status" >> "$TIMING_CSV"
    echo ""
    echo "================================================================"
    echo "TOTAL training time: $total_hms ($total s)  status=$status"
    echo "   per-cell timings logged to: $TIMING_CSV"
    echo "================================================================"
}
trap on_exit EXIT

# --- helper ----------------------------------------------------------------
# Args:  out_name  dataset_root  network  cell  [extra augmentation flags...]
train() {
    local out_name=$1
    local dataset_root=$2
    local network=$3
    local cell=$4
    shift 4
    local extras=("$@")

    # --compile policy: skip for dynunet (heavy graph) and for any AFA cell
    # (dual-norm route switching). The train script also auto-skips, but
    # gating here keeps the log clean.
    local compile_flag="--compile"
    if [[ "$network" == "dynunet" ]] || [[ " ${extras[*]} " == *" --use_afa "* ]]; then
        compile_flag=""
    fi

    # Pre-create the run directory so we can tee the log into it.
    local out_dir="./runs/${out_name}"
    mkdir -p "$out_dir"
    local log_file="$out_dir/train.log"

    # Modality derived from out_name (first underscore-separated token, e.g.
    # 'BTFE_dynunet_regular' → 'BTFE'). Used only for the timing CSV row.
    local modality="${out_name%%_*}"

    local cell_start
    cell_start=$(date +%s)
    local cell_start_iso
    cell_start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "==========================================================="
    echo "TRAIN  net=$network  mod=$out_name  cell=$cell"
    echo "  log: $log_file"
    echo "  tail in another terminal:  tail -f \"$log_file\""
    echo "  started: $cell_start_iso"
    echo "==========================================================="

    # Run python with set -e temporarily off so we can capture the exit code
    # for the timing CSV (pipefail still propagates a python crash through tee).
    local status="ok"
    set +e
    python -u train_placenta_2d_monai_v8.py \
        --dataset_root  "$dataset_root" \
        --out_dir       "$out_dir" \
        --mode          train \
        --network       "$network" \
        --cv_folds      1 \
        --epochs        "$EPOCHS" \
        --batch_size    "$BATCH" \
        --lr            "$LR" \
        --weight_decay  "$WEIGHT_DECAY" \
        --scheduler     plateau \
        --early_stopping --patience "$PATIENCE" \
        --num_workers   "$WORKERS" \
        --amp $compile_flag $RESUME_FLAG \
        --seed          "$SEED" \
        "${extras[@]}" \
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

    # Append to the global timing CSV (one row per cell)
    echo "$cell_start_iso,$cell_end_iso,$duration,$hms,train,$network,$modality,$cell,$out_dir,$status" >> "$TIMING_CSV"

    # Also stamp the duration into the cell's own log
    echo "$out_name: $hms ($duration s)  status=$status" | tee -a "$log_file"

    # Propagate a real failure (don't silently continue past a crashed cell)
    if [ "$status" = "failed" ]; then
        echo "ERROR: Cell $out_name failed (exit code $exit_code) — see $log_file"
        exit "$exit_code"
    fi
}

# --- modality -> dataset root resolver ------------------------------------
modality_root() {
    case $1 in
        BTFE) echo "$BTFE_ROOT" ;;
        TSE)  echo "$TSE_ROOT"  ;;
        *)    echo "UNKNOWN_MODALITY"; exit 1 ;;
    esac
}

# --- the four augmentation cells (per the AFA paper composition rule) -----
# Cell flags:
#   none              : --aug_level none
#   regular           : --aug_level regular
#   afa_mixup         : --aug_level regular --use_afa --use_mixup --mixup_alpha 0.2
#   afa_cutmix_mixup  : --aug_level regular --use_afa --use_mixup --use_cutmix
#                       --mixup_alpha 0.2 --cutmix_alpha 1.0 --mixup_prob 0.5
#                       (per-batch 50/50 one-of MixUp or CutMix, then AFA aux path)
cell_flags() {
    case $1 in
        none)
            echo "--aug_level none"
            ;;
        regular)
            echo "--aug_level regular"
            ;;
        afa_mixup)
            echo "--aug_level regular --use_afa --use_mixup --mixup_alpha 0.2"
            ;;
        afa_cutmix_mixup)
            echo "--aug_level regular --use_afa --use_mixup --use_cutmix --mixup_alpha 0.2 --cutmix_alpha 1.0 --mixup_prob 0.5"
            ;;
        *)
            echo "UNKNOWN_CELL"; exit 1 ;;
    esac
}

CELLS=("none" "regular" "afa_mixup" "afa_cutmix_mixup")

# --- the matrix: network × modality × cell --------------------------------
for NETWORK in "${NETWORKS[@]}"; do
    for MOD in "${MODALITIES[@]}"; do
        ROOT=$(modality_root "$MOD")
        for CELL in "${CELLS[@]}"; do
            OUT_NAME="${MOD}_${NETWORK}_${CELL}"
            FLAGS=$(cell_flags "$CELL")
            # shellcheck disable=SC2086
            train "$OUT_NAME" "$ROOT" "$NETWORK" "$CELL" $FLAGS
        done
    done
done

echo ""
echo "================================================================"
echo "All training runs complete. Trained models are saved under ./runs/"
echo "================================================================"
