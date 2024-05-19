#!/bin/bash

# 设置参数值
num_epochs=600
mode="test"
pre_types="pre" 
model_types="U_Net"
cuda_idx=2
loss_types=("Dice" "BCE" )
best_model_path="/checkpoints/U_Net_pre_epoch251.pth"

# 循环运行main.py，通过命令行参数传递不同的值
for pre_type in "${pre_types[@]}"; do
    for model_type in "${model_types[@]}"; do
        for loss_type in "${loss_types[@]}"; do
            eval python main.py \
                --model_type "$model_type" \
                --pre_type "$pre_type" \
                --cuda_idx "$cuda_idx" \
                --mode "$mode" \
                --num_epochs "$num_epochs" \
                --best_model_path "$best_model_path" \
                --loss_type "$loss_type"  # 添加 loss_type 参数
        done
    done
done
