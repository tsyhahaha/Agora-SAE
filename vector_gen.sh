export CUDA_VISIBLE_DEVICES=1

python -m agora_sae.scripts.generate_activations \
    --preset math500-1.5b \
    --model /home/taosiyuan/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --layer 27 \
    --reasoning-datasets /mnt/data1/taosiyuan/Agora-SAE/sampled_data \
    --output ./data/math500_activations/layer_27 \
    --batch-size 32 \
    --buffer-size-mb 500 \
    --shard-size-mb 100