#!/usr/bin/env bash

# Function to check if GPUs are available
check_gpu_availability() {
    local gpus=$1
    local threshold=1500  # 1500MB threshold
    
    for gpu in $(echo $gpus | tr ',' ' '); do
        local used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)
        if [ $used_memory -gt $threshold ]; then
            echo "GPU $gpu is busy (used memory: $used_memory MB)"
            return 1
        fi
    done
    return 0
}

# Function to wait for GPUs to become available
wait_for_gpus() {
    local gpus=$1
    echo "Waiting for GPUs ($gpus) to become available..."
    while ! check_gpu_availability "$gpus"; do
        echo "GPUs are busy, waiting for 30 seconds..."
        sleep 30
    done
    echo "GPUs are now available!"
}

# =============================================================================
# ==================== 1. 通用配置 =====================

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 判断 GPU 数量，自动选择单卡或多卡模式
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS: $NUM_GPUS"
wait_for_gpus "$CUDA_VISIBLE_DEVICES"


export TMPDIR='/data1/yuanqianguang/tmp'

# bash dist_train.sh /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/mycelium_model/model/deeplabv3_r50-d8.py 4 "/data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/work_dirs/deeplabv3_r50-d8"

# bash dist_train.sh /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/mycelium_model/model/pspnet_unet_s5-d16.py 4 "/data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/work_dirs/pspnet_unet_s5-d16"

bash dist_train.sh /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/mycelium_model/model/segformer.b0.1024x1024.city.160k.py 4 "/data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/work_dirs/segformerb0"

