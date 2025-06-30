bs=8
lr=0.015
epochs=1000

model='unet'

dir_img='/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/images_GT'
dir_mask='/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/'
# save_checkpoint_path='./checkpoints_unet/metal_1000/only_metal_1000'
# save_checkpoint_path='./checkpoints_unet/cera_800/only_cera_800'
# save_checkpoint_path='./checkpoints_curve/metal_1000_0.015'
save_checkpoint_path='./checkpoints_curve/cera_800_0.015'

CUDA_VISIBLE_DEVICES=2 python ../train.py \
    --num_workers 2 \
    --epochs=$epochs \
    --learning_rate=$lr \
    --batch_size=$bs \
    --model=$model \
    --model_path=$save_checkpoint_path \
    --tb_path ./ \
    --softmax \
    --cos \
    --dir_img=$dir_img \
    --dir_mask=$dir_mask 