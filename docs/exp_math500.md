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

### 1.3 SAE 架构与超参数

当前仓库现成可跑通的是 `Top-K SAE + layer 12` 版本；但如果目标是**严格贴近论文复现主线**，则应当以论文 `Section 4.2` 的设置为准。

**当前仓库实现**:
- **目标层 (Hook Point)**: `layers.12.residual_stream` (中后期层，偏逻辑表达)
- **模型维度 ($d_{model}$)**: 1,536
- **膨胀系数 (Expansion Factor)**: 32x
- **字典维度 ($d_{sae}$)**: 49,152
- **稀疏度 ($k$)**: 32 (Top-32 激活)
- **学习率 (Learning Rate)**: 5e-5 (配合 Warmup)
- **Batch Size (Offline Trainer)**: 4096 / 8192 (取决于截断后的单分片内存大小)

**论文 target setting**:
- **SAE 类型**: 标准 SAE，而不是 Top-K SAE
- **目标维度 ($D$)**: 2,048
- **Batch Size**: 1,024
- **Learning Rate**: `1e-4`
- **Warmup**: 前 10%
- **稀疏强度**: `lambda = 2e-3`
- **主分析层位**: final layer

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

论文复现主线的评估重点不是 `PPL` 或通用 feature 频率统计，而是回答下面三个问题：
- SAE decoder columns 是否在几何上对应到可区分的 reasoning behaviors。
- 这些 behaviors 是否能被人工或 judge 稳定标注为 `reflection / backtracking / other`。
- 把对应的 behavior vector 注入回原模型后，能否因果性地改变推理风格，同时尽量保持答案正确。

因此，`math500` 复现的 Phase 3 应以论文 `Section 4.3` 和 `Section 4.4.1` 为核心，而不是以当前仓库中的通用 `evaluate_sae` 脚本为核心。后者最多只能算补充诊断，不属于论文主评估流程。

#### Step 3.1: 行为标注 (Behavior Labeling)

对 `MATH500` 中每个 reasoning step 对应的 step-level activation，按论文设定标成以下三类：
- `reflection`: 回看并重新检查前面步骤。
- `backtracking`: 放弃当前路线，切换到另一种解法或分支。
- `other`: 不属于以上两类的其余 step。

论文这里采用的是 `LLM-as-a-judge`，并明确使用了 `GPT-5`。一阶段复现时应保留这一点，至少要保留“外部 judge 标注 reasoning step”这个核心流程。

**执行命令**:
```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

python -m agora_sae.scripts.evaluate_paper_math500 label-steps \
    --dataset-path <LOCAL_MATH500_PATH> \
    --output ./eval/math500_step_labels.jsonl \
    --response-source model \
    --model <LOCAL_MODEL_PATH> \
    --judge openai \
    --judge-model gpt-5 \
    --max-samples 500 \
    --max-new-tokens 512
```

**说明**:
- 这条命令会先让目标模型对题目生成 response，再按 `\n\n` 切 step，最后对每个 step 做 judge 标注。
- 如果你只是想先跑通本地流程，可以临时改成 `--judge heuristic`；但那不属于论文主线复现。

**复现产物**:
- 一份逐 step 标注后的表格或 JSONL。
- 每条记录至少包含：`sample_id`、`step_id`、`step_text`、`label`。

#### Step 3.2: SAE Decoder Geometry 分析

在拿到逐 step 标签之后，论文主线做两件事：
- 找出各类行为对应的 top-active channels / decoder columns。
- 对 decoder columns 做 UMAP 投影，查看 `reflection`、`backtracking`、`other` 是否在几何上呈现可分离结构。

论文还补充了 layer-wise 的 silhouette score 分析，用来量化不同层的行为可分离性；其中 later layers 通常更清晰。

**一阶段复现的最低要求**:
- 至少完成 final layer 上的 decoder column 可视化。
- 至少能展示 `reflection` 和 `backtracking` 两类在 decoder space 中不是完全混在一起。

**执行命令**:
```bash
python -m agora_sae.scripts.evaluate_paper_math500 analyze-geometry \
    --labels ./eval/math500_step_labels.jsonl \
    --checkpoint ./checkpoints/math500-1.5b/checkpoint_final.pt \
    --model <LOCAL_MODEL_PATH> \
    --layer 12 \
    --output-dir ./eval/geometry_math500 \
    --embedding-method umap \
    --plot-path ./eval/geometry_math500/decoder_umap.png
```

**说明**:
- 如果你已经有 final-layer SAE checkpoint，这里应把 `--layer` 改成对应 final layer。
- `--embedding-method umap` 需要环境里安装 `umap-learn`；如果只是想先做降级版检查，可以改成 `pca`。

**复现验收点**:
- 能输出一张 decoder column 的二维投影图。
- 图上 `reflection`、`backtracking` 对应的高分 columns 有可见聚类趋势。

#### Step 3.3: 因果干预 (Causal Intervention)

这是论文主评估里最关键的一步。流程是：
- 从 final layer 中筛出 behavior-specific decoder columns。
- 对同一类列向量取平均，得到一个 `reflection vector` 或 `backtracking vector`。
- 在原模型推理时，把该向量注入到“每个 reasoning step 的最后一个 token”对应的隐藏状态。
- 比较 `negative / vanilla / positive` 三种条件下，reflection 或 backtracking 的 step 数是否按预期变化。

论文的核心观察不是 `PPL`，而是：
- 推理风格是否随 intervention 强度稳定变化。
- 最终答案是否尽量保持不变。
- 这种 effect 是否能跨任务泛化。

**一阶段复现的最低要求**:
- 先在 `R1-1.5B + MATH500` 上完成 final-layer intervention。
- 至少比较三种条件：`negative`、`vanilla`、`positive`。
- 统计每种条件下的 `# reflection steps` / `# backtracking steps` 和最终答案正确率。

**执行命令**:
```bash
python -m agora_sae.scripts.evaluate_paper_math500 run-intervention \
    --dataset-path <LOCAL_MATH500_PATH> \
    --geometry-summary ./eval/geometry_math500/geometry_summary.json \
    --checkpoint ./checkpoints/math500-1.5b/checkpoint_final.pt \
    --model <LOCAL_MODEL_PATH> \
    --layer 12 \
    --behavior reflection \
    --output ./eval/intervention_reflection.jsonl \
    --judge openai \
    --judge-model gpt-5 \
    --conditions negative:1.0,vanilla:0.0,positive:-1.0 \
    --max-samples 32 \
    --max-new-tokens 384
```

**说明**:
- 这一步会从 geometry summary 里挑出目标行为的 top decoder columns，平均成一个 behavior vector，再对原模型做 `negative / vanilla / positive` 三种干预。
- 当前命令默认使用 `reflection` 做示例；如果你要复现实验里的 `backtracking`，只需改 `--behavior backtracking`。

**复现产物**:
- 一张类似论文 Figure 5 / Figure 6 的结果汇总表。
- 每个条件下至少记录：`task_id`、`condition`、`# reflection steps`、`# backtracking steps`、`final_answer`、`is_correct`。

#### Step 3.4: 跨任务泛化

如果要继续贴近论文主线，下一步不是做通用 PPL，而是把在 `MATH500` 学到的 behavior vectors 拿到其他任务上验证。

论文正文里列出的方向包括：
- `AIME 2025`
- `AMC23`
- 以及跨领域的 `GPQA-Diamond`、`KnowLogic`

对一阶段复现来说，这一步可以放在 `MATH500` 内部因果干预之后，但仍然属于论文主线，不属于工程附加项。

**当前仓库状态**:
- 目前同一套 intervention 脚本已经能复用于别的数据集，但还没有现成的任务封装模板。

#### 当前脚本的定位

当前仓库里的 [evaluate_sae.py](/Users/siyuantao/repos/Agora-SAE/agora_sae/scripts/evaluate_sae.py) 仍然可以保留，但它不应再被写成 `math500` 论文复现的主评估命令。论文主线请优先使用新的 [evaluate_paper_math500.py](/Users/siyuantao/repos/Agora-SAE/agora_sae/scripts/evaluate_paper_math500.py)。

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
3. **行为标注可落地**: 至少完成 `reflection / backtracking / other` 的逐 step 标注，并能回溯到对应激活点。
4. **Decoder Geometry 可解释**: final layer 上，`reflection` 与 `backtracking` 对应的高分 decoder columns 在 UMAP 空间中应表现出可见结构，而不是完全混杂。
5. **因果干预有效**: 对 behavior vector 做 `negative / vanilla / positive` 干预后，相关行为步数应当按预期变化。
6. **答案尽量保持稳定**: 干预主要改变 reasoning style，而不是简单破坏求解能力；至少在一组代表性数学题上，最终答案应尽量保持一致或只出现轻微波动。
