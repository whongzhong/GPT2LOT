#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/userhome/whzhong/code/GPT2LOT
export WANDB_MODE=offline
export LC_ALL=C.UTF-8

exec 1>info/normal.output
exec 2>info/normal.error

python main.py --do_test \
    --test_model 'BART-epoch=18.ckpt' \
    --model_name 'BART' \
    --test_batch_size 7 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --output_dir 'output/permute' \
    --model_path "data/models/BART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/permute' 

    
python main.py --do_test \
    --test_model 'BART-epoch=49.ckpt' \
    --model_name 'BART' \
    --test_batch_size 6 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --output_dir 'output/cbart/normalfp' \
    --model_path "data/models/BART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/cbart/normalfp' 
    #--data_root './LOTdatasets/outgen/board/data'
    
python main.py --do_test \
    --test_model 'BART-epoch=39.ckpt' \
    --model_name 'BART' \
    --test_batch_size 10 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --output_dir 'output/cbart/quotation' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_dir 'ckpts/cbart/quotation' 
    #--data_root './LOTdatasets/outgen/board/data'
    
/userhome/anaconda3/envs/lot10/bin/python main.py --cont_train\
    --epoch_num 40 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '512' \
    --root '/userhome/whzhong/code/GPT2LOT' \
    --model_path "data/models/CBART" \
    --data_root 'data/datasets/LOTdatasets' \
    --ckpt_load_dir 'ckpts/cbart' \
    --test_model 'BART-epoch=36.ckpt'
    
    
    

    