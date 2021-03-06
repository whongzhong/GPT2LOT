#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/userhome/whzhong/code/GPT2LOT
export WANDB_MODE=offline
export LC_ALL=C.UTF-8

exec 1>info/cbartgen.output
exec 2>info/cbartgen.error

/userhome/anaconda3/envs/lot10/bin/python main.py --do_test \
    --test_model 'BART-epoch=36.ckpt' \
    --model_name 'BART' \
    --test_batch_size 32 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --output_dir 'output/cbart' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/cbart' 