#!/usr/bin/env bash
# Seg-3: fine-tune the Mendeley-trained seg models on the 5 anomaly-train Rebro
# patients (reb003/006/009/013/014) and test on the 7 held-out Rebro
# (reb001/004/005/007/008/010/011). Runs BOTH TSE_dynunet_afa_mixup and
# TSE_dynunet_regular so we get a side-by-side before/after.
#
# Split: DATASET_REBRO_FT/splits.json (train=5, val=[], test=7); cv_folds=1 ->
# a single 80/20 of the 5 (4 train / 1 val) for early stopping, matching how the
# base models were trained. Transfer via --checkpoint (warm start), low LR 1e-4.
set -uo pipefail
cd "$(dirname "$0")"
PY="python -u"   # run inside the activated Python environment (see requirements.txt)
FT=dataset/mri_png/DATASET_REBRO_FT
TRAIN_COMMON="--dataset_root $FT --mode train --network dynunet --cv_folds 1 \
  --epochs 50 --batch_size 16 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler plateau --early_stopping --patience 15 \
  --num_workers 20 --amp --seed 42 --aug_level regular"

run_one () {
  local name=$1 base_ckpt=$2 train_extra=$3 test_extra=$4
  local out=runs/${name}_rebroFT
  echo "################## FINE-TUNE  $name  ##################"
  $PY train_placenta_2d_monai_v8.py $TRAIN_COMMON --out_dir "$out" \
      --checkpoint "$base_ckpt" $train_extra || { echo "TRAIN FAILED $name"; return 1; }
  echo "################## TEST  $name  on 7 held-out ##################"
  $PY train_placenta_2d_monai_v8.py --dataset_root "$FT" --mode test \
      --network dynunet --out_dir "$out/test_on_heldout" \
      --checkpoint "$out/fold_0/best_model.pth" $test_extra --amp \
      || { echo "TEST FAILED $name"; return 1; }
}

run_one TSE_dynunet_afa_mixup runs/TSE_dynunet_afa_mixup/fold_0/best_model.pth \
        "--use_afa --use_mixup --mixup_alpha 0.2" "--use_afa"
run_one TSE_dynunet_regular   runs/TSE_dynunet_regular/fold_0/best_model.pth \
        "" ""
echo "SEG3_ALL_DONE"
