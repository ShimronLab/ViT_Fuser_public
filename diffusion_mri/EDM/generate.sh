CUDA_VISIBLE_DEVICES=1
NPROC=1
ROOT=../
# DPS
MODEL_PATH=/home/taloved/tal-lxc/tal/docker/gsure-diffusion-mri/models/edm/brain/32dB/00018-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-007536.pkl
# DPS PI
MODEL_PATH=/home/taloved/tal-lxc/tal/docker/gsure-diffusion-mri/models/edm/brain/32dB_PI/00006-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-000460.pkl

SAMPLE_DIM=172,108
NUM_SAMPLES=10
BATCH_SIZE=10

ANATOMY=brain
DATA=noisy32dB

torchrun --standalone --nproc_per_node=$NPROC generate.py \
        --outdir=$ROOT/results/priors/$ANATOMY/$DATA --seeds=1-$NUM_SAMPLES \
        --batch=$BATCH_SIZE --network=$MODEL_PATH \
        --sample_dim=$SAMPLE_DIM --gpu=$CUDA_VISIBLE_DEVICES