CKPT_PATH='output/stage1-256-single-vangogh-new/checkpoints/model_20001.pkl'

CUDA_VISIBLE_DEVICES=2 python src/StyDeco_inference.py \
    --model_path=$CKPT_PATH \
    --root_path "eval_data/vangogh2photo/B_all" \
    --prompt "eval_data/vangogh2photo/captionB.json" --direction "b2a" \
    --output_dir "/data/lxy/sqj/code/img2img-turbo/my_samples/stage1-256-single-vangogh-20001/B_output" \
    --image_prep "resize_256x256"
