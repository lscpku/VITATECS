CUDA_VISIBLE_DEVICES=2 python eval_vitatecs_xclip.py --do_eval \
    --output_dir ckpts/xclip_msrvtt_vit32_mpool \
    --max_words 32 --max_frames 12 --batch_size_val 64 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name ViT-B/32 \
    --init_model ckpts/xclip_msrvtt_vit32_mpool/pytorch_model.bin.4

CUDA_VISIBLE_DEVICES=2 python eval_vitatecs_clip4clip.py --do_eval \
    --output_dir ckpts/clip4clip_msrvtt_vit32_mpool \
    --max_words 32 --max_frames 12 --batch_size_val 64 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name ViT-B/32 \
    --init_model ckpts/clip4clip_msrvtt_vit32_mpool/pytorch_model.bin.4