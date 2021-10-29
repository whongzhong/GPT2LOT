#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/userhome/whzhong/code/GPT2LOT
export WANDB_MODE=offline
export LC_ALL=C.UTF-8

exec 1>info/normal.output
exec 2>info/normal.error

/userhome/anaconda3/envs/lot10/bin/python main.py \
    --epoch_num 30 \
    --train_batch_size 16 \
    --val_batch_size 30 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '400' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/BART" \
    --data_root 'data/datasets/LOTdatasets/permute_data' \
    --ckpt_dir 'ckpts/permute' \
    --group_name 'permute_train' 
    # --data_root 'data/datasets/essaydatasets/essay_only' \
    #--test_model 'BART-epoch=19.ckpt'
    #--test_model 'BART-epoch=19-v1.ckpt' 
    #--data_root './LOTdatasets/extradata' \
    #--data_root './LOTdatasets/outgen/board/data'