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
- **Sampled Dataset**: `MATH500`
  - **Local Path**: `<LOCAL_MATH500_PATH>` (由下文的采样脚本生成)

### 1.2 数据采样与处理逻辑 (Data Ingestion)
- **MATH500 抽样**:
  - 从全量 MATH 数据集中，设置固定的 Random Seed，**随机抽取 500 条数据**作为本次实验的子集（`MATH500`）。
- **Step 级切割与激活取点**:
  - 针对这 500 条数据，**保留完整的 Query 与最终 reasoning / Solution 作为模型上下文输入**，不破坏原始推理上下文。
  - **核心逻辑**：先把 reasoning 部分按 `\n\n` 切成 step，再只在每个 step 的 delimiter 位置提取目标层激活；若最后一个 step 后没有显式 delimiter，则回退到该 step 的末 token，确保每条样本至少贡献一个 step-level activation point。
  - **解释**：这样做把“怎么切 step”和“在哪个点取 activation”拆开了，后续可以换成别的规则甚至 LLM judge，而不需要重写激活生成主流程。

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

### Phase 0: 生成本地 `MATH500` 子集

先从本地全量数学数据集中抽取固定 500 条，保证后续实验可复现。

**执行命令**:
```bash
python -m agora_sae.scripts.sample_dataset \
    --dataset-path <LOCAL_DATASET_PATH> \
    --output-path <LOCAL_MATH500_PATH> \
    --num-samples 500 \
    --seed 42
```

**参数解释**:
- `--dataset-path <LOCAL_DATASET_PATH>`: 本地 clone 或保存下来的原始数学数据集目录。
- `--output-path <LOCAL_MATH500_PATH>`: 采样后生成的新数据集目录。
- `--num-samples 500`: 抽样数量，默认就是 500。
- `--seed 42`: 固定随机种子，便于复现实验。

### Phase 1: 离线激活值生成 (Activation Generation)

从目标大模型中提取特定层的激活向量。`math500-1.5b` preset 会保留完整 Query 作为上下文输入，并按 `\n\n` 划分 reasoning step，只保存每个 step 的 delimiter activation。

**执行命令**:
```bash
python -m agora_sae.scripts.generate_activations \
    --preset math500-1.5b \
    --model <LOCAL_MODEL_PATH> \
    --reasoning-datasets <LOCAL_MATH500_PATH> \
    --output ./data/math500_activations \
    --batch-size 16 \
    --buffer-size-mb 500 \
    --shard-size-mb 100
```
**参数解释**:
- `--preset math500-1.5b`: 使用此实验专属的预设配置（系统会自动设置好目标网络层数为第 12 层等）。
- `--model <LOCAL_MODEL_PATH>`: 替换为你下载到本地的模型权重文件夹路径。
- `--reasoning-datasets <LOCAL_MATH500_PATH>`: 指向 Phase 0 生成的 500 条本地子集目录。
- `--output ./data/math500_activations`: 所提取出来的特征向量（`.safetensors` 文件）最后要保存到的目标文件夹。
- `--batch-size 16`: 推荐先从 16 开始；如果显存足够，可以提高到 32。
- `--buffer-size-mb 500`: 在把特征写入磁盘之前，会在内存里积攒并打乱（Shuffle）特征；这个设置决定了用于打乱的缓冲区大小（单位：MB）。设置得越大，训练数据随机性越好，但内存占用也更高。
- `--shard-size-mb 100`: 保存到硬盘上的每个特征数据小切片（Shard）的最大体积限制（单位：MB）。
- `--max-batches`: 可选，只用于 smoke test；默认会在本地有限数据集遍历完一轮后自动停止，不会无限循环。

- **数据流向**: `MixedTokenSource` 会加载本地数据集，保留完整 Query 上下文，先对 reasoning 片段做 step 切分，再用 activation point selector 只标记各个 step delimiter 对应的 token 位置。然后注入 `OfflineActivationGenerator`，前向传播捕获第 12 层输入侧 residual stream 的激活张量。
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

如果暂时不想启用 Weights & Biases，可以使用下面这条可直接运行的命令：

```bash
python -m agora_sae.scripts.train_sae \
    --preset math500-1.5b \
    --shards ./data/math500_activations \
    --checkpoint-dir ./checkpoints/math500-1.5b \
    --batch-size 4096 \
    --lr 5e-5 \
    --steps 100000 \
    --no-wandb
```

### Phase 3: 特征解释与验证 (Evaluation)

验证稀疏网络是否能“听懂”人类的数学意图。主要使用预留的验证集（或者手动 prompt 测试）。
- 抽取特征字典中的 Top-10 / Top-20 最频发活跃特征，还原到文本切片中观察。预期能捕捉到数学步骤特定的语义概念，例如：
  - 特征 A: 专门在 `\n\n` 切断后的独立条件陈述句激活 (e.g., "Let $x$ be...")。
  - 特征 B: 在推断转折与纠错句激活 (e.g., "Wait, this is incorrect...").
  - 特征 C: 在最终计算化简步骤激活。

**执行命令**:
```bash
python -m agora_sae.scripts.evaluate_sae \
    --checkpoint ./checkpoints/math500-1.5b/checkpoint_final.pt \
    --model <LOCAL_MODEL_PATH> \
    --layer 12 \
    --shards ./data/math500_activations \
    --skip-ppl
```

**参数解释**:
- `--checkpoint`: Phase 2 训练完成后生成的最终 SAE checkpoint。
- `--model <LOCAL_MODEL_PATH>`: 本地模型权重路径。
- `--layer 12`: 对应本实验使用的目标层。
- `--shards ./data/math500_activations`: 可用于后续特征利用率分析。
- `--skip-ppl`: 如果当前环境没有准备额外评测数据，先跳过 PPL 测试，保证命令可直接运行。

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
- **磁盘占用 (Disk)**: 当前只保存 step delimiter activation，而不是整段 token activation。对 500 条数学题，通常只会留下几千到几万条 step-level 向量；按 1536 维 bf16 粗估，落盘体积通常在 **`数十 MB 到数百 MB`** 量级，明显低于之前的 token-span 方案。
- **系统内存 (RAM)**: 内存洗牌队列 (Buffer) 配置为 `500 MB`，加上数据加载进程的开销，系统内存需求仅需 `< 8 GB`。

---

## 4. 预期结果 (Acceptance Criteria)

1. **成功产出 MATH500 激活文件**: 在 `./data/math500_activations` 目录上生成有效的 `.safetensors` 分片，且分片内的激活对应于 reasoning step delimiter 的 token 位置，而不是整段 token span。
2. **训练收敛**: Wandb 监控可见 `L2 Reconstruct Error` 平稳下降至预期水平（L2 Ratio $< 5\%$），且由于 $k=32$，L0 激活特征严格固定在 `32`。
3. **低死神经元率 (Dead Latents)**: 通过设置 `Aux Loss Weight`=`1/32`，确保在 MATH500 特定领域的子集内，特征死亡率处于极低水平 (`< 1%`)。
