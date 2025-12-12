#!/bin/bash
set -e

prefix="data/visium_mouse_brain_ct_hd/"

# Toggle supervision / fake spots for training
use_ct_supervision=true
use_fake_hd_spots=true
hd_loss_weight=0.1
ct_loss_weight=1.0
data_dir="/gpfs/commons/home/svaidyanathan/istarc/data"  # base directory for heat diffused data

impute_flags=()
if [ "$use_ct_supervision" = "true" ]; then
  impute_flags+=(--use-ct-supervision)
fi
if [ "$use_fake_hd_spots" = "true" ]; then
  impute_flags+=(--use-fake-hd-spots)
fi
impute_flags+=(--hd-loss-weight="${hd_loss_weight}")
impute_flags+=(--ct-loss-weight="${ct_loss_weight}")
impute_flags+=(--data-dir="${data_dir}")

# run pipeline
./run.sh "$prefix" "${impute_flags[@]}"
