export CUDA_VISIBLE_DEVICES=1

python -m agora_sae.scripts.train_sae \
    --preset math500-1.5b \
    --layer-range 0:24 \
    --layer-step 4 \
    --final-layer 27 \
    --shards ./data/math500_activations \
    --checkpoint-dir ./checkpoints/math500-layer-scan \
    --batch-size 4096 \
    --lr 5e-5 \
    --steps 100000 \
    --no-wandb