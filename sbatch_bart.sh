#!/bin/bash
#SBATCH --error error_bart.out                       # 输出错误
#SBATCH -J bart-gen                               # 作业名为 test
#SBATCH -o test_bart.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                    # 单任务使用的 CPU 核心数为 4
#SBATCH -t 30:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1

source ~/.bashrc

# 设置运行环境
conda activate lot

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python main.py \
    --epoch_num 30 \
    --train_batch_size 14 \
    --val_batch_size 30 \
    --eos_token '[EOS]' \
    --bos_token '[BOS]' \
    --delimeter_token '<DELIMETER>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'BART' \
    --max_length '400' \
    --model_path "models/BART"\
    --data_root './LOTdatasets/datasets/essaydatasets' \
    --ckpt_dir 'ckpts_max_data' 
    #--test_model 'BART-epoch=19-v1.ckpt' 
    #--data_root './LOTdatasets/extradata' \
    #--data_root './LOTdatasets/outgen/board/data'