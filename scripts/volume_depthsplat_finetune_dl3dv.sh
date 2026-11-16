#!/usr/bin/env bash
# DepthSplat on volume-rendered scenes, experiment 1: fine-tuned from the base DL3DV model.
#
# Starts from depthsplat-gs-base-dl3dv-256x448-randview2-6 (weights only; the optimizer, the learning-rate schedule
# and the step count start fresh). Uses experiment "volume" as is: 256x256, 6 context + 4 target views (even view
# sampling), cameras normalized per scene (volume center at the origin, bounding ball of radius k), tight near / far
# around that ball, the DL3DV model's encoder settings, upstream loss (MSE + 0.05 LPIPS) and optimizer.
#
# Run from the repo root. Settings are environment variables, for example
#   ROOTS=/path/CQ500_merged_split,/path/nyx_merged_split BATCH_SIZE=1 bash scripts/volume_depthsplat_finetune_dl3dv.sh
# Extra Hydra overrides can be appended (keys not set below), e.g. wandb.mode=disabled or wandb.entity=<account>.
# Multi-node: append trainer.num_nodes=<N> (every node runs this script; all GPUs of a node are used).
set -euo pipefail

# training roots, comma-separated; each holds train/ and test/ with <scene>/gaussian_splat/
ROOTS=${ROOTS:-datasets/CQ500_merged_split,datasets/nyx_012_00215_055_lambert_merged_split,datasets/nekrs_tgv_velocity_magnitude_merged_split}
CKPT=${CKPT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}
# per-GPU batch; measured on one GPU (fp32, 6 + 4 views): 11.4 GB at 1 for forward + backward, 2 did not fit in 24 GB.
# Upstream's DL3DV fine-tune also used 1 per GPU.
BATCH_SIZE=${BATCH_SIZE:-1}
# one epoch is (training scenes) / (BATCH_SIZE x GPUs) steps
MAX_STEPS=${MAX_STEPS:-100001}
LR=${LR:-2e-4}
LR_MONODEPTH=${LR_MONODEPTH:-2e-6}
OUT=${OUT:-checkpoints/volume-256x256-depthsplat-base-finetune-dl3dv}

python -m src.main +experiment=volume \
    "dataset.roots=[${ROOTS}]" \
    data_loader.train.batch_size="${BATCH_SIZE}" \
    trainer.max_steps="${MAX_STEPS}" \
    optimizer.lr="${LR}" \
    optimizer.lr_monodepth="${LR_MONODEPTH}" \
    checkpointing.pretrained_model="${CKPT}" \
    wandb.project=depthsplat \
    output_dir="${OUT}" \
    "$@"

# To resume this run after an interruption, run the script again with checkpointing.resume=true appended: it picks
# up the latest checkpoint under <OUT>/checkpoints (optimizer and step count included).
#
# Evaluate a checkpoint from the FFGS-benchmark repo root with the same k and bounds:
#   python -m exps.eval_depthsplat --variant base-dl3dv --bounds tight --k 1.0 \
#       --ckpt methods/depthsplat/<OUT>/checkpoints/<name>.ckpt \
#       --root <test root> --eval_json <eval json> --out_dir <out dir>
