
export NCCL_P2P_DISABLE=1
LR="1e-5"
PROJECT_NAME="style_vangogh"
EPOCH=40000
BATCH_SIZE=1
VAL_STEP=1000
RANK_TEXT_ENCODER=16


CUDA_VISIBLE_DEVICES=1 \
    python src/StyDeco_trainer.py \
    --pretrained_model_name_or_path="/data/lxy/sqj/base_models/sd-turbo" \
    --output_dir="outputs/$PROJECT_NAME" \
    --dataset_folder "examples/vangogh" \
    --train_img_prep "resize_286_randomcrop_256x256_hflip" \
    --learning_rate=$LR --max_train_steps=$EPOCH \
    --train_batch_size=$BATCH_SIZE --gradient_accumulation_steps=2 \
    --tracker_project_name=$PROJECT_NAME \
    --lora_rank_text_encoder=$RANK_TEXT_ENCODER \
    --validation_steps=$VAL_STEP --checkpointing_steps=$VAL_STEP \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 \
    --enable_xformers_memory_efficient_attention \
    --report_to "wandb"
