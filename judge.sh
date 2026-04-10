export CUDA_VISIBLE_DEVICES=1

python -m agora_sae.scripts.evaluate_paper_math500 label-steps \
    --dataset-path /mnt/data1/taosiyuan/Agora-SAE/sampled_data \
    --output ./eval/math500_step_labels.jsonl \
    --response-source model \
    --model /home/taosiyuan/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-samples 500 \
    --max-new-tokens 512 \
    --judge minimax \
    --judge-model MiniMax-M2.5 \
    --minimax-max-output-tokens 384 \
    --judge-timeout 90 \
    --judge-max-retries 5 \
    --resume