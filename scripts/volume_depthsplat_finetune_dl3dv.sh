#!/usr/bin/env bash


# base model, fine-tuned on volume-rendered scenes (256x256, 6 context + 4 target views, even view sampling)
# starts from the base DL3DV model; config/experiment/volume.yaml holds that model's encoder settings
#
# each dataset root holds train/ and test/, with <scene>/gaussian_splat/{transforms.json, render_config.json, images/}
# cameras are normalized per scene: volume center at the origin, bounding ball of radius dataset.k,
# near / far tight around that ball (dataset.bounds=tight)
#
# batch size 1 on each gpu: 6 + 4 views take about 17GB (fp32)
# one epoch is (number of training scenes) / (batch size x number of gpus) steps: set trainer.max_steps from the
# number of epochs wanted
# add wandb.mode=disabled to log locally
python -m src.main +experiment=volume \
data_loader.train.batch_size=1 \
dataset.roots=[datasets/CQ500_merged_split,datasets/nyx_012_00215_055_lambert_merged_split,datasets/nekrs_tgv_velocity_magnitude_merged_split] \
dataset.k=1.0 \
dataset.bounds=tight \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.num_target_views=4 \
trainer.max_steps=100000 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
wandb.project=depthsplat \
output_dir=checkpoints/volume-256x256-depthsplat-base-view6


# evaluate a checkpoint from the FFGS-benchmark repo root, with the same k and bounds:
# python -m exps.eval_depthsplat --variant base-dl3dv --bounds tight --k 1.0 \
#     --ckpt methods/depthsplat/checkpoints/volume-256x256-depthsplat-base-view6/checkpoints/<name>.ckpt \
#     --root <test root> --eval_json <eval json> --out_dir <out dir>
