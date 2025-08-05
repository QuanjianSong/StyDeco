PRETRAINED_PATH="/data/lxy/sqj/base_models/sd-turbo"
CKPT_PATH='outputs/style_vangogh/checkpoints/model_1.pkl'


# b->a
CUDA_VISIBLE_DEVICES=2 python src/StyDeco_inference.py \
    --pretrained_path=$PRETRAINED_PATH \
    --model_path=$CKPT_PATH \
    --root_path "examples/vangogh/eval_B" \
    --prompt "examples/vangogh/caption_eval_B.json" --direction "b2a" \
    --output_dir "samples/style_vangogh/B_output" \
    --image_prep "resize_256x256"

# a->b
CUDA_VISIBLE_DEVICES=2 python src/StyDeco_inference.py \
    --pretrained_path=$PRETRAINED_PATH \
    --model_path=$CKPT_PATH \
    --root_path "examples/vangogh/eval_A" \
    --prompt "examples/vangogh/caption_eval_A.json" --direction "a2b" \
    --output_dir "samples/style_vangogh/A_output" \
    --image_prep "resize_256x256"
