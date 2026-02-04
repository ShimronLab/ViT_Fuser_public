CUDA_VISIBLE_DEVICES=0
NPROC=1
ANATOMY=brain
NATIVE_SNR=32dB
ROOT=../
MODEL_PATH=/home/taloved/tal-lxc/tal/docker/gsure-diffusion-mri/models/edm/brain/32dB_halbach/00023-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl
STEPS=500

# 2. Update Sample Range (We only generated sample_0.pt)
SAMPLE_START=0
SAMPLE_END=1

DATA=noisy32dB
INFERENCE_SNR=32dB
R=3
SEED=15


MEAS_PATH=../inference_data/measurements
KSP_PATH=../inference_data/kspace

torchrun --standalone --nproc_per_node=$NPROC dps.py \
    --seed $SEED --latent_seeds $SEED --gpu=$CUDA_VISIBLE_DEVICES \
    --sample_start $SAMPLE_START --sample_end $SAMPLE_END \
    --inference_R $R --inference_snr $INFERENCE_SNR \
    --num_steps $STEPS --S_churn 0 \
    --measurements_path $MEAS_PATH \
    --ksp_path $KSP_PATH \
    --network=$MODEL_PATH \
    --outdir=$ROOT/results/posterior/$ANATOMY/$DATA\
    --snr 32