---

# Document A: SAE 训练系统工程开发规范 (Code Agent Spec)

**Project Name**: `DeepSeek-SAE-Offline-Reasoning`
**Version**: 2.0 (Staged Offline Pipeline)
**Target Stack**: PyTorch, TransformerLens, Safetensors, FlashAttention
**Architecture**: Top-K SAE with Auxiliary Loss & Dead Latent Recovery

---

## 1. 系统架构概览 (System Architecture)

本项目采用 **Staged Offline (分阶段离线)** 架构，以解耦 LLM 推理与 SAE 训练，最大化显存利用率。

### 模块划分

1. **Module A: Data Ingestion & Formatting** (数据清洗与混合)
2. **Module B: Activation Generator** (Stage 1: 离线激活值生成)
3. **Module C: SAE Modeling** (Top-K SAE 核心模型)
4. **Module D: Offline Trainer** (Stage 2: 高吞吐训练器)
5. **Module E: Analysis & Evaluation** (验证与可视化)

---

## 2. 详细模块规范 (Detailed Specs)

### Module A: Data Ingestion (数据摄入)

**目标**: 构建符合 `80% Reasoning / 20% General` 配比的数据流。

* **Class**: `MixedTokenSource`
* **Inputs**:
* `reasoning_datasets`: List[str] (e.g., "MATH500", "OpenR1-Math")
* `general_datasets`: List[str] (e.g., "FineWeb-Edu")
* `ratio`: float = 0.8


* **Logic**:
* 实现 `IterableDataset`。
* **Reasoning Parser**: 针对 Reasoning 数据，编写提取器。
* **Strict Filter & Split**: 必须保留完整的 `Query`、`<think>` 内部文本以及 `Solution` 作为推理上下文输入模型。在**提取目标激活值**时，则针对于 `<think>` 和 `Solution` 部分的推导链，**强制以 `\n\n` 为界限对推理过程进行逻辑分段**，仅针对分段内的 Token 提取并保存激活向量 (sentence-level extraction)。通过这种方式，在保留模型完整语义背景的同时，实现细粒度的特征对齐与字典学习。
* **Tokenization**: 使用 `AutoTokenizer` (Qwen/DeepSeek)。
* **Output**: Stream of `input_ids` (tensor).





### Module B: Activation Generator (Stage 1)

**目标**: 运行 LLM，提取目标层激活，保存为分片文件。

* **Class**: `OfflineActivationGenerator`
* **Config**:
* `model_name`: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
* `hook_point`: "layers.12.input" (根据模型调整)
* `shard_size`: int = 100MB (per file)
* `storage_path`: "./buffer_shards/"


* **Method `run_generation_loop()**`:
* 加载 LLM (BF16, Inference Mode, `device_map="auto"`).
* 设置 Batch Size = Max allowed (e.g., 64/128).
* 注册 Hook 提取 `residual_stream`。
* **Global Shuffle Buffer**: 在内存中维护一个 500MB 的 Buffer。
* 写入数据 -> 当 Buffer 满时 -> Shuffle -> 写入磁盘 `.safetensors`。


* **Rolling Mechanism**: 监控磁盘使用量，如果超过阈值 (e.g., 200GB)，暂停生成，等待训练器消费。





### Module C: SAE Modeling (模型架构)

**目标**: 实现带 Auxiliary Loss 的 Top-K SAE。

* **Class**: `TopKSAE(nn.Module)`
* **Init Params**:
* `d_model`: int (1536 for 1.5B)
* `d_sae`: int (49152, i.e., **32x Expansion**)
* `k`: int (32 ~ 64)


* **Components**:
* `W_enc`: `[d_model, d_sae]`, bias `b_enc`
* `W_dec`: `[d_sae, d_model]`, bias `b_dec`


* **Forward**:
1. **Pre-Norm**: 
2. **Encode**: 
3. **Top-K**: `vals, inds = torch.topk(z, k)`
4. **Decode**: 


* **Loss Calculation**:
* `L_recon`: MSE()
* `L_aux`: **Dead Latent Reconstruction**.
* 计算残差 。
* 针对本 Batch 未被激活的 Latents (Dead Set)，尝试用它们预测 。
* `aux_loss = MSE(e, dead_latents_prediction)`。
* *Note*: 这是一个 Auxiliary Head，不更新 `W_enc` 主梯度，只更新 Dead Latents 的方向。




* **Constraints**:
* `@torch.no_grad` method `remove_decoder_gradients_wrt_active_latents()` (Optional, for advanced TopK).
* `@torch.no_grad` method `set_decoder_norm()`: **Force Unit Norm** on `W_dec` columns.





### Module D: Offline Trainer (Stage 2)

**目标**: 高吞吐消费磁盘上的激活值进行训练。

* **Class**: `SAETrainer`
* **Config**:
* `lr`: 5e-5
* `batch_size`: 4096 (maximize VRAM)
* `steps`: Total ~100k (covering 400M tokens)


* **DataLoader**:
* 自定义 `ShardLoader`：多线程读取 `.safetensors` 分片。
* 支持 **Delete-after-Read** (消费完即删除分片，配合 Rolling Buffer)。


* **Training Loop**:
1. Load Batch .
2. Forward SAE -> Loss = .
3. Backward -> Optimizer Step.
4. **Constraint Application**: Call `set_decoder_norm()`.
5. **Dead Latent Check**: 每 2500 steps 统计一次 Dead Latents。如果比例 > 10%，触发 Anthropic-style Resampling (可选，优先依赖 Aux Loss)。


* **Logging**: 使用 `wandb` 记录 `L2`, `L0` (Effective K), `ExplainedVariance`.



### Module E: Analysis (分析模块)

**目标**: 验证特征有效性。

* **Script**: `eval_sae.py`
* **Reconstruction Test**:
* 加载原始 LLM。
* 替换目标层为 SAE (Reconstruction)。
* 在 `Wikitext-2` 和 `GSM8K` 上测 Perplexity / Accuracy。
* **Acceptance Criteria**: PPL 增加 < 5%。


* **Feature Browsing**:
* 输入一段推理文本 (CoT)。
* 提取 Top-10 激活特征。
* 打印特征编号及激活强度，人工检查是否对应具体逻辑步骤 (e.g., "Calculation", "Summarization")。
