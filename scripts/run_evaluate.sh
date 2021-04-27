#!/bin/bash


test_step=true
eval_step=true

ckpt_dir=exp
gpus=0

if $test_step; then
    python -B ./test.py \
        --gpu_ids=$gpus \
        --tt_list=../filelists/tt_list.txt \
        --ckpt_dir=$ckpt_dir \
        --model_file=./${ckpt_dir}/models/best.pt
fi

if $eval_step; then
    python -B ./measure.py --metric=stoi --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
    python -B ./measure.py --metric=pesq --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
    python -B ./measure.py --metric=snr --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
fi
