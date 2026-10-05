#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
PY="python -u"   # run inside the activated Python environment (see requirements.txt)
FT=dataset/mri_png/DATASET_REBRO_FT
out=runs/TSE_dynunet_afa_mixup_rebroFT
echo "##### FINE-TUNE afa_mixup (fixed) #####"
$PY train_placenta_2d_monai_v8.py --dataset_root $FT --mode train --network dynunet --cv_folds 1 \
  --epochs 50 --batch_size 16 --lr 1e-4 --weight_decay 1e-5 --scheduler plateau \
  --early_stopping --patience 15 --num_workers 20 --amp --seed 42 --aug_level regular \
  --out_dir $out --checkpoint runs/TSE_dynunet_afa_mixup/fold_0/best_model.pth \
  --use_afa --use_mixup --mixup_alpha 0.2 || { echo AFA_TRAIN_FAILED; exit 1; }
echo "##### TEST afa_mixup on 7 held-out #####"
$PY train_placenta_2d_monai_v8.py --dataset_root $FT --mode test --network dynunet \
  --out_dir $out/test_on_heldout --checkpoint $out/fold_0/best_model.pth --use_afa --amp \
  || { echo AFA_TEST_FAILED; exit 1; }
echo "AFA_DONE"
