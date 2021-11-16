#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/userhome/whzhong/code/GPT2LOT
export WANDB_MODE=offline
export LC_ALL=C.UTF-8


exec 1>info/cbart_newtokencls.out
exec 2>info/cbart_newtokencls.error 

# run for cbart normal
/userhome/anaconda3/envs/lot10/bin/python main.py \
    --epoch_num 50 \
    --train_batch_size 8 \
    --val_batch_size 30 \
    --eos_token '[SEP]' \
    --bos_token '[CLS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/cbart/new_token/cls' \
    --group_name 'cbart_new_token_cls' 
    
    
    
/userhome/anaconda3/envs/lot10/bin/python main.py --do_test \
    --test_model 'BART-epoch=40.ckpt' \
    --model_name 'BART' \
    --test_batch_size 30 \
    --eos_token '[SEP]' \
    --bos_token '[CLS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/cbart/new_token/cls' \
    --output_dir 'output/new_token/cls' 
    
/userhome/anaconda3/envs/lot10/bin/python utils/eval.py output/new_token/cls/BART_test.json output/test.jsonl
    
/userhome/anaconda3/envs/lot10/bin/python main.py --do_test \
    --test_model 'BART-epoch=49.ckpt' \
    --model_name 'BART' \
    --test_batch_size 30 \
    --eos_token '[SEP]' \
    --bos_token '[CLS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/cbart/new_token/cls' \
    --output_dir 'output/new_token/cls/49' 

/userhome/anaconda3/envs/lot10/bin/python utils/eval.py output/new_token/cls/49/BART_test.json output/test.jsonl
