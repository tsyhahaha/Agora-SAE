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

### Stage 1: 生成激活值

```bash
# 使用预设配置
agora-generate --preset deepseek-1.5b --output ./buffer_shards/

# 自定义配置
agora-generate \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --layer 12 \
    --output ./buffer_shards/ \
    --batch-size 64
```

### Stage 2: 训练 SAE

```bash
# 使用预设配置
agora-train --preset deepseek-1.5b --shards ./buffer_shards/

# 自定义配置
agora-train \
    --shards ./buffer_shards/ \
    --d-model 1536 \
    --expansion 32 \
    --k 32 \
    --batch-size 4096 \
    --steps 100000
```

### 评估模型

```bash
agora-eval \
    --checkpoint ./checkpoints/checkpoint_final.pt \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --layer 12
```

## Python API

```python
from agora_sae import TopKSAE, SAETrainer, Config

# 创建配置
config = Config.from_preset("deepseek-1.5b")

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

## License

MIT License
