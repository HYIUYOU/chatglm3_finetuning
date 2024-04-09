#!/bin/bash


OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/root/heyiyuan/project/ChatGLM3/ChatGLM-Finetuning/output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
    --train_path /root/heyiyuan/project/ChatGLM3/dataset/dataset.json \
    --model_name_or_path /root/heyiyuan/project/ChatGLM3/chatglm3-6b/ \
    --per_device_train_batch_size 1 \
    --max_len 512 \
    --max_src_len 256 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.1 \
    --mode glm3 \
    --train_type freeze \
    --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
    --seed 1234 \
    --ds_file /root/heyiyuan/project/ChatGLM3/ChatGLM-Finetuning/ds_zero3_no_offload.json \
    --gradient_checkpointing \
    --show_loss_step 10 \
    --output_dir $OUTPUT \
    &> $OUTPUT/training_raw_data.log

