#!/bin/bash

# 设置参数值
num_epochs=600
pre_types=("pre" "pre1" "pre2" "pre3" "pre4" "pre5" "pre6" "pre7" "pre14" "pre16" "pre25" "pre27")
model_types=("U_Net" "R2U_Net" "AttU_Net" "R2AttU_Net")
cuda_idx=0
loss_types=("mixed" )
best_model_path="best_model.pth"

# 循环运行main.py，通过命令行参数传递不同的值
for pre_type in "${pre_types[@]}"; do
    for model_type in "${model_types[@]}"; do
        for loss_type in "${loss_types[@]}"; do
            eval python main.py \
                --model_type "$model_type" \
                --pre_type "$pre_type" \
                --cuda_idx "$cuda_idx" \
                --num_epochs "$num_epochs" \
                --loss_type "$loss_type"  # 添加 loss_type 参数
        done
    done
done
