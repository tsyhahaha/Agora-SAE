export CUDA_VISIBLE_DEVICES=1

for layer in 0 4 8 12 16 20 24 27; do
  python -m agora_sae.scripts.generate_activations \
      --preset math500-1.5b \
      --model /home/taosiyuan/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
      --layer ${layer} \
      --reasoning-datasets /mnt/data1/taosiyuan/Agora-SAE/sampled_data \
      --output ./data/math500_activations/layer_${layer} \
      --batch-size 16 \
      --buffer-size-mb 500 \
      --shard-size-mb 100
done