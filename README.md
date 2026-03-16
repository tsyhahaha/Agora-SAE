# Agora-SAE

**Staged Offline SAE Training System for Reasoning Model Interpretability**

基于 Top-K Sparse Autoencoder 的推理模型可解释性分析工具。

## Installation

```bash
# 基础安装
pip install -e .

# 包含开发依赖
pip install -e ".[dev]"

# 包含 Flash Attention（可选）
pip install -e ".[flash]"
```

## Quick Start

### 1. 准备一个 500 条的本地数学子集

```bash
python -m agora_sae.scripts.sample_dataset \
    --dataset-path /path/to/competition_math \
    --output-path /path/to/competition_math_500 \
    --num-samples 500 \
    --seed 42
```

### 2. Stage 1: 生成激活值

```bash
python -m agora_sae.scripts.generate_activations \
    --preset math500-1.5b \
    --model /path/to/local/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-datasets /path/to/competition_math_500 \
    --output ./data/math500_activations \
    --batch-size 16 \
    --buffer-size-mb 500 \
    --shard-size-mb 100
```

### 3. Stage 2: 训练 SAE

```bash
python -m agora_sae.scripts.train_sae \
    --preset math500-1.5b \
    --shards ./data/math500_activations \
    --checkpoint-dir ./checkpoints/math500-1.5b \
    --batch-size 4096 \
    --lr 5e-5 \
    --steps 100000 \
    --wandb-project agora-sae \
    --wandb-run math500-1.5b-run1
```

### 4. 评估模型

```bash
python -m agora_sae.scripts.evaluate_sae \
    --checkpoint ./checkpoints/math500-1.5b/checkpoint_final.pt \
    --model /path/to/local/DeepSeek-R1-Distill-Qwen-1.5B \
    --layer 12 \
    --shards ./data/math500_activations \
    --skip-ppl
```

说明：
- `generate_activations` 现在支持 Hugging Face dataset 名称和本地 dataset 路径两种输入。
- `math500-1.5b` preset 会保留完整 query 作为上下文输入，并只在 reasoning step 的分隔点上提取激活。
- 对有限本地数据集，`generate_activations` 默认单轮跑完后自动结束；只有显式加 `--repeat-data` 才会循环重跑。
- 如果 `--batch-size 16` 显存还有余量，可以再提高到 `32`。
- 如果你把项目安装成包，也可以直接使用 `agora-sample-dataset`、`agora-generate`、`agora-train`、`agora-eval`。

### 常用自定义命令

```bash
agora-generate \
    --model /path/to/local/model \
    --reasoning-datasets /path/to/local/dataset \
    --reasoning-ratio 1.0 \
    --output ./buffer_shards \
    --batch-size 16

agora-train \
    --preset deepseek-1.5b \
    --shards ./buffer_shards \
    --checkpoint-dir ./checkpoints/run1 \
    --batch-size 4096 \
    --steps 100000
```

## Python API

```python
from agora_sae import TopKSAE, SAETrainer
from agora_sae.config import get_config

# 创建配置
config = get_config("deepseek-1.5b")

# 创建模型
sae = TopKSAE(
    d_model=config.model.d_model,
    d_sae=config.d_sae,
    k=config.sae.k
)

# 训练
trainer = SAETrainer(sae, config)
trainer.train(shard_loader)
```

## Project Structure

```
Agora-SAE/
├── agora_sae/              # 主包
│   ├── data/               # 数据处理
│   ├── activation/         # 激活生成
│   ├── model/              # SAE 模型
│   ├── trainer/            # 训练器
│   ├── eval/               # 评估工具
│   └── scripts/            # CLI 脚本
├── tests/                  # 单元测试
├── docs/                   # 文档
└── pyproject.toml          # 项目配置
```

## Configuration Presets

| Preset | Model | Layer | d_model | d_sae | k |
|--------|-------|-------|---------|-------|---|
| `deepseek-1.5b` | DeepSeek-R1-Distill-Qwen-1.5B | 12 | 1,536 | 49,152 | 32 |
| `qwen3-8b` | Qwen/Qwen3-8B | 16 | 4,096 | 131,072 | 64 |
| `qwq-32b` | Qwen/QwQ-32B | 24 | 5,120 | 163,840 | 64 |
| `math500-1.5b` | DeepSeek-R1-Distill-Qwen-1.5B | 12 | 1,536 | 49,152 | 32 |

## License

MIT License
