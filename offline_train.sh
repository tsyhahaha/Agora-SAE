export CUDA_VISIBLE_DEVICES=1

python -m agora_sae.scripts.train_sae \
    --preset math500-1.5b \
    --layer 27 \
    --shards ./data/math500_activations/layer_27 \
    --checkpoint-dir ./checkpoints/math500-final-layer \
    --batch-size 4096 \
    --lr 5e-5 \
    --steps 100000 \
    --no-wandb