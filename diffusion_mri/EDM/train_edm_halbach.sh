CUDA_VISIBLE_DEVICES=0
NPROC=4
LOADER=Noisy
ANATOMY=brain
DATA=noisy
SNR=32dB_halbach
ROOT=../
ROOT_DATA='/home/taloved/tal-lxc/tal/docker/gsure-diffusion-mri/data/brain/train/32dB_halbach/noisy.pt'
BATCH_SIZE=128
NORMALIZE=0
PRECOND=edm
AUGMENT=1


torchrun --standalone --nproc_per_node=$NPROC train.py \
 --outdir=$ROOT/models/$PRECOND/$ANATOMY/$SNR \
 --data=$ROOT_DATA \
 --cond=0 --arch=ddpmpp --duration=10 \
 --batch=$BATCH_SIZE --cbase=128 --cres=1,1,2,2,2,2,2 \
 --lr=1e-4 --ema=0.1 --dropout=0.0 \
 --desc=container_test --tick=1 --snap=10 \
 --dump=200 --seed=2023 --precond=$PRECOND --augment=$AUGMENT \
 --normalize=$NORMALIZE --loader=$LOADER --gpu=$CUDA_VISIBLE_DEVICES