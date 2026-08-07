#!/bin/bash
# Copies selected plots from the toy_probe_hiding sweep output into result/.
#
# To add a plot: find its filename in
#   ~/ml/toy_probe_hiding/publish/sweep3_lam0.1_tr0/best
# and add a new line to the FILES list below, WITHOUT the common
# "sweep3_lam0.1_tr0_" prefix (it gets added automatically).
#
# Run this script from anywhere: ./copy_plots.sh

set -e

DEST_DIR="$(dirname "$0")/result"
trash "$DEST_DIR"
mkdir -p "$DEST_DIR"

SRC_DIR=~/ml/toy_probe_hiding/publish/sweep7_lam0.1_tr0/best
PREFIX="sweep7_lam0.1_tr0_"
FILES=(
  auroc_bar.png
  curves.png
  linear_y.png
  L2_pca.png
  L2_probe.png
  L2_steer_dir_mag.png
  L4_steer_cmp.png
  roc_noise_grid.png
)
for f in "${FILES[@]}"; do
  cp "$SRC_DIR/$PREFIX$f" "$DEST_DIR/$PREFIX$f"
done


SRC_DIR=~/ml/toy_probe_hiding/publish/sweep7_lam0_tr0/best
PREFIX="sweep7_lam0_tr0_"
FILES=(
  L2_steer_dir_mag.png
)
for f in "${FILES[@]}"; do
  cp "$SRC_DIR/$PREFIX$f" "$DEST_DIR/$PREFIX$f"
done

echo "Copied files to $DEST_DIR"
