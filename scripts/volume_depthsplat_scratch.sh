#!/usr/bin/env bash
# DepthSplat on volume-rendered scenes, experiment 2: trained from scratch (no DepthSplat checkpoint).
#
# Same data, views, normalization, tight bounds, encoder settings (base: ViT-B), loss and optimizer as experiment 1
# (scripts/volume_depthsplat_finetune_dl3dv.sh). What "from scratch" starts from is set by INIT:
#   INIT=backbones  (default) DepthSplat's own from-scratch recipe, as in its README and re10k training script: the
#                   depth backbone starts from Depth Anything V2 (ViT-B) and UniMatch / GMFlow weights, everything else
#                   (Gaussian heads, feature upsampler) is random. Needs both files in pretrained/:
#                     wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -P pretrained
#                     wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
#   INIT=random     no pretrained weights at all: the monodepth ViT is built without DINOv2's weights
#                   (model.encoder.monodepth_vit_pretrained=false) and every parameter uses one learning rate
#                   (optimizer.lr_monodepth=0.0), since the 2e-6 rate upstream uses for the pretrained ViT would barely
#                   train a random one.
#
# Run from the repo root. Settings are environment variables, for example
#   ROOTS=/path/CQ500_merged_split,/path/nyx_merged_split INIT=backbones bash scripts/volume_depthsplat_scratch.sh
# Extra Hydra overrides can be appended (keys not set below), e.g. wandb.mode=disabled or wandb.entity=<account>.
# Multi-node: append trainer.num_nodes=<N> (every node runs this script; all GPUs of a node are used).
set -euo pipefail

# training roots, comma-separated; each holds train/ and test/ with <scene>/gaussian_splat/
ROOTS=${ROOTS:-datasets/CQ500_merged_split,datasets/nyx_012_00215_055_lambert_merged_split,datasets/nekrs_tgv_velocity_magnitude_merged_split}
INIT=${INIT:-backbones}
# per-GPU batch; measured on one GPU (fp32, 6 + 4 views): 11.4 GB at 1 for forward + backward, 2 did not fit in 24 GB
BATCH_SIZE=${BATCH_SIZE:-1}
# one epoch is (training scenes) / (BATCH_SIZE x GPUs) steps; from scratch likely needs more steps than fine-tuning
MAX_STEPS=${MAX_STEPS:-100001}
LR=${LR:-2e-4}
OUT=${OUT:-checkpoints/volume-256x256-depthsplat-base-scratch-${INIT}}

case "${INIT}" in
    backbones)
        INIT_ARGS=(checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth
                   checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth
                   optimizer.lr_monodepth="${LR_MONODEPTH:-2e-6}")
        ;;
    random)
        INIT_ARGS=(model.encoder.monodepth_vit_pretrained=false optimizer.lr_monodepth=0.0)
        ;;
    *)
        echo "INIT must be backbones or random, not '${INIT}'" >&2
        exit 1
        ;;
esac

python -m src.main +experiment=volume \
    "dataset.roots=[${ROOTS}]" \
    data_loader.train.batch_size="${BATCH_SIZE}" \
    trainer.max_steps="${MAX_STEPS}" \
    optimizer.lr="${LR}" \
    "${INIT_ARGS[@]}" \
    wandb.project=depthsplat \
    output_dir="${OUT}" \
    "$@"

# To resume this run after an interruption, run the script again (same INIT) with checkpointing.resume=true appended:
# it picks up the latest checkpoint under <OUT>/checkpoints (optimizer and step count included).
#
# Evaluate a checkpoint from the FFGS-benchmark repo root with the same k and bounds (the checkpoint holds every
# encoder weight, so the wrapper's DINOv2 initialization is overwritten, also for INIT=random):
#   python -m exps.eval_depthsplat --variant base-dl3dv --bounds tight --k 1.0 \
#       --ckpt methods/depthsplat/<OUT>/checkpoints/<name>.ckpt \
#       --root <test root> --eval_json <eval json> --out_dir <out dir>
