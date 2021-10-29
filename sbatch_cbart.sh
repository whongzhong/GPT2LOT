#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/userhome/whzhong/code/GPT2LOT
export WANDB_MODE=offline
export LC_ALL=C.UTF-8

exec 1>info/cbart_shuffle.out
exec 2>info/cbart_shuffle.error 

# run for cbart permute
/userhome/anaconda3/envs/lot10/bin/python main.py \
    --epoch_num 40 \
    --train_batch_size 16 \
    --val_batch_size 48 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets/permute_data' \
    --ckpt_dir 'ckpts/cbart/permute' \
    --group_name 'cbart_permute' 
    
# run for cbart rerake
/userhome/anaconda3/envs/lot10/bin/python main.py \
    --epoch_num 40 \
    --train_batch_size 16 \
    --val_batch_size 48 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets/rerake_agument' \
    --ckpt_dir 'ckpts/cbart/rerake' \
    --group_name 'cbart_rerake'