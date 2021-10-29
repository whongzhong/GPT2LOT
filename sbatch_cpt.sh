#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/userhome/whzhong/code/GPT2LOT
export WANDB_MODE=offline
export LC_ALL=C.UTF-8

exec 1>info/cpt30.out
exec 2>info/cpt30.error 

/userhome/anaconda3/envs/lot10/bin/python main.py --cont_train \
    --epoch_num 30 \
    --train_batch_size 20 \
    --val_batch_size 48 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'CPT' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/CPT" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/cpt/60' \
    --ckpt_load_dir 'ckpts/cpt' \
    --test_model 'CPT-epoch=29.ckpt'\
    --group_name 'cpt' 