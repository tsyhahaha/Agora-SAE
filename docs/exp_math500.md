# 实验文档：MATH500 上的 SAE 训练 (Experiment 1)

**实验代号**：`Exp-01-MATH500-1.5B`
**目标**：在小规模数学推理数据集 (MATH500) 上走通 SAE (Sparse Autoencoder) 的离线提取与训练全流程，验证特定模型下的数据处理切割逻辑与特征提取有效性。

---

## 1. 实验核心配置

### 1.1 模型与数据集路径
- **Target Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - **Local Path**: `<LOCAL_MODEL_PATH>` (替换为你下载到本地的模型权重路径)
- **Source Dataset**: `MATH`
  - **Local Path**: `<LOCAL_DATASET_PATH>` (替换为你下载到本地的数据集路径)

### 1.2 数据采样与处理逻辑 (Data Ingestion)
- **MATH500 抽样**:
  - 从全量 MATH 数据集中，设置固定的 Random Seed，**随机抽取 500 条数据**作为本次实验的子集（`MATH500`）。
- **细粒度切割 (Sentence-level Split for Extraction)**:
  - 针对这 500 条数据，**必须保留完整的 Query, `<think>` (CoT) 及最终 Solution 作为模型的上下文输入**，不能破坏模型推理状态。
  - **核心逻辑**：在**提取目标层激活值**时，重点关注 `<think>` 和最终 `Solution` 部分，将其**严格以 `\n\n` 为界限划分为逻辑分段**，并仅提取/保存各分段内部的激活向量。
  - **解释**：这种提取策略（保留全加上下文，但分割提取范围）有效迫使 SAE 将关注点缩窄至特定的单步逻辑操作上（如“展开多项式”、“计算判别式”），避免因提取超长连续片段带来的特征混淆，以捕捉更高质量、更具解释性的微观特征。

### 1.3 SAE 架构与超参数 (Top-K SAE)
基于 `1.5B` 模型的参数量规模与内存规划：
- **目标层 (Hook Point)**: `layers.12.residual_stream` (中后期层，偏逻辑表达)
- **模型维度 ($d_{model}$)**: 1,536
- **膨胀系数 (Expansion Factor)**: 32x
- **字典维度 ($d_{sae}$)**: 49,152
- **稀疏度 ($k$)**: 32 (Top-32 激活)
- **学习率 (Learning Rate)**: 5e-5 (配合 Warmup)
- **Batch Size (Offline Trainer)**: 4096 / 8192 (取决于截断后的单分片内存大小)

---

## 2. 实验执行流程与具体命令

本实验遵循项目中定义的 **Staged Offline** 架构，分为激活生成、SAE 训练和特征分析三个阶段。

### Phase 1: 离线激活值生成 (Activation Generation)

从目标大模型中提取特定层的激活向量，由于使用的是 `math500-1.5b` preset，系统会自动保证仅包含推理数据集，保留 Query，并按 `\n\n` 进行截断。

**执行命令**:
```bash
python -m agora_sae.scripts.generate_activations \
    --preset math500-1.5b \
    --model <LOCAL_MODEL_PATH> \
    --reasoning-datasets <LOCAL_DATASET_PATH> \
    --output ./data/math500_activations \
    --batch-size 32 \
    --buffer-size-mb 500 \
    --shard-size-mb 100
```
**参数解释**:
- `--preset math500-1.5b`: 使用此实验专属的预设配置（系统会自动设置好目标网络层数为第 12 层等）。
- `--model <LOCAL_MODEL_PATH>`: 替换为你下载到本地的模型权重文件夹路径。
- `--reasoning-datasets <LOCAL_DATASET_PATH>`: 替换为你本地的数学数据集路径。
- `--output ./data/math500_activations`: 所提取出来的特征向量（`.safetensors` 文件）最后要保存到的目标文件夹。
- `--batch-size 32`: 每次喂给大模型进行前向推理计算的样本数量。如果显存不足（OOM），可以适当把这个数字调小（比如 16 或 8）。
- `--buffer-size-mb 500`: 在把特征写入磁盘之前，会在内存里积攒并打乱（Shuffle）特征；这个设置决定了用于打乱的缓冲区大小（单位：MB）。设置得越大，训练数据随机性越好，但内存占用也更高。
- `--shard-size-mb 100`: 保存到硬盘上的每个特征数据小切片（Shard）的最大体积限制（单位：MB）。

- **数据流向**: `MixedTokenSource` 会加载数据集，经过 Parser 按预设的 `\n\n` 进行分段切割。然后注入 `OfflineActivationGenerator`，前向传播捕获第 12 层 (`layers.12.residual_stream`) 的激活张量。
- **落地文件**: 生成的内容会被保存在 `./data/math500_activations` 下的多个紧凑的 `.safetensors` 分片文件中。

### Phase 2: 高吞吐训练 (Offline SAE Training)

消耗存储在磁盘上的激活文件，通过 Top-K 稀疏自编码器拟合特征字典。

**执行命令**:
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
**参数解释**:
- `--preset math500-1.5b`: 沿用和提取阶段相同的预设配置，以确保自动使用对应的特征维度（例如 32 倍膨胀）和 Top-K（如 K=32）。
- `--shards ./data/math500_activations`: 指向上一个阶段提取并保存激活特征分片文件的那个文件夹。
- `--checkpoint-dir ./checkpoints/math500-1.5b`: SAE 模型自身训练完毕（或中途保存）时，其权重的存储位置目录。
- `--batch-size 4096`: SAE 字典模型每次更新权重时“看”多少个特征标记（Tokens）。仅仅训练这个小模型而不涉及大模型语言推理，显存非常充裕，可以设置得大一些（如 4096 甚至 8192）。
- `--lr 5e-5`: 学习率（Learning Rate），控制 SAE 每次通过数据学习并更新字典权重的步幅大小。
- `--steps 100000`: SAE 训练过程要进行的优化步数，即总共跑多少次批次（Batch）的数据。
- `--wandb-project`: 在 Weights & Biases 监控系统记录指标时所用的项目名称。（由于你是本地运行，如果不需要监控记录面板，可以在命令参数中附加 `--no-wandb` 将其关闭）。
- `--wandb-run`: 在给定的追踪项目里，本实验的具体展示名称。

- **数据流向**: `InfiniteShardLoader` (或 `ShardLoader`) 异步加载分片到内存打乱。训练主循环将其按 Batch 喂给 `TopKSAE` 模型。
- **核心约束**: 损失函数主要由 MSE 重建损失，辅以死亡神经元抑制 (Aux Loss) 构成。内部会严格对 Decoder 权重应用 L2 Unit Norm 约束。

### Phase 3: 特征解释与验证 (Evaluation)

验证稀疏网络是否能“听懂”人类的数学意图。主要使用预留的验证集（或者手动 prompt 测试）。
- 抽取特征字典中的 Top-10 / Top-20 最频发活跃特征，还原到文本切片中观察。预期能捕捉到数学步骤特定的语义概念，例如：
  - 特征 A: 专门在 `\n\n` 切断后的独立条件陈述句激活 (e.g., "Let $x$ be...")。
  - 特征 B: 在推断转折与纠错句激活 (e.g., "Wait, this is incorrect...").
  - 特征 C: 在最终计算化简步骤激活。

---

## 3. 计算资源需求估算

在此预设规模下 (`Qwen-1.5B`, $d_{model}=1536$, Expansion$=32x$), 资源消耗预估如下：

### 3.1 算力与显存 (VRAM)
- **阶段一 (生成激活)**:
  - 模型本身 (1.5B 参数，bf16精度) 占用约 ~3 GB。
  - 推理时 KV Cache 和中间激活张量（Batch Size=32, Seq Len=2048）占用约 6-8 GB。
  - **总显存需求**: 约 `12 GB`。单张 RTX 3090/4090 (24GB) 或主流服务器 GPU 足以跑满。
- **阶段二 (SAE 训练)**:
  - SAE 参数量：Encoder 权重 ($1536 \times 49152$) + Decoder 权重 ($49152 \times 1536$) 约 150M 参数。FP32 存储 + Adam 优化器状态，占用约 `1.8 GB` 显存。
  - 训练 Batch 数据占用极小 (Batch=4096，1536 维，FP32) < 100 MB。
  - **总显存需求**: `< 4 GB`，极度轻量。

### 3.2 存储与内存 (RAM/Disk)
- **磁盘占用 (Disk)**: 500 条数学题（平均 1000-2000 tokens），总计约 `1e6` tokens。每个 Token 的激活向量 (1536维 * 2 Bytes = 3 KB)。预计生成激活总文件大小在 **`3 GB`以内**。磁盘空间压力很小。
- **系统内存 (RAM)**: 内存洗牌队列 (Buffer) 配置为 `500 MB`，加上数据加载进程的开销，系统内存需求仅需 `< 8 GB`。

---

## 4. 预期结果 (Acceptance Criteria)

1. **成功产出 MATH500 激活文件**: 在 `./data/math500_activations` 目录上生成有效的 `.safetensors` 分片，且由于使用了 `math500-1.5b` 预设，分片内的激活对应于按照 `\n\n` 细粒度逻辑切断后的文本片段。
2. **训练收敛**: Wandb 监控可见 `L2 Reconstruct Error` 平稳下降至预期水平（L2 Ratio $< 5\%$），且由于 $k=32$，L0 激活特征严格固定在 `32`。
3. **低死神经元率 (Dead Latents)**: 通过设置 `Aux Loss Weight`=`1/32`，确保在 MATH500 特定领域的子集内，特征死亡率处于极低水平 (`< 1%`)。
