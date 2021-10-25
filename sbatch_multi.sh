#!/bin/bash
#SBATCH --error error_gpu2.out                       # 输出错误
#SBATCH -J multi_gpu                               # 作业名为 test
#SBATCH -o test_gpu2.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=16                    # 单任务使用的 CPU 核心数必须为4
#SBATCH -t 12:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:4

source ~/.bashrc

# 设置运行环境
conda activate lot

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=.

python main.py \
    --epoch_num 12 \
    --train_batch_size 4 \
    --val_batch_size 16 \
    --eos_token '<eod>' \
    --bos_token '<bos>' \
    --delimeter_token '<sep>' \
    --sep_token '<sep>' \
    --pad_token '[PAD]' \
    --model_name 'CPM' \
    --max_length '512' \
    --model_path "mymusise/CPM-Generate-distill"\
    --data_root './LOTdatasets/outgen/board/data'