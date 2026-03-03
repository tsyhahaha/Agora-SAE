---

# 稀疏自编码器 (SAE) 全流程复现与分析文档

**版本**：v2.0 (Staged Offline Pipeline)

**项目代号**：Project Agora (DeepSeek-SAE-Offline-Reasoning)

**目标模型**：DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-8B, QwQ-32B

**核心方法**：Top-K SAE (with Auxiliary Loss) & Offline Dictionary Learning

---

## 第一部分：理论基础与数学框架 (Theoretical Foundations)

本部分涵盖工程实现所依赖的核心理论，确保开发者理解“为什么要这样写代码”。

### 1.1 核心问题：叠加态 (Superposition) 与多义性

LLM 的残差流 (Residual Stream) 维度 $d_{model}$ (如 1536 或 4096) 远小于模型实际掌握的概念数量。根据 Johnson-Lindenstrauss 引理和 **Superposition Hypothesis**，模型通过将特征以非正交的方式“压缩”在低维空间中。

- **工程推论**：直接分析 $d_{model}$ 维度的激活向量是无意义的，因为存在多义性 (Polysemanticity)。必须将其解压到高维空间 $d_{sae}$ ($d_{sae} \gg d_{model}$)。

### 1.2 解决方案：稀疏字典学习 (Sparse Dictionary Learning)

SAE 将激活向量 $x$ 分解为一组稀疏特征的线性组合：

$$
x \approx \sum_{i \in \text{active}} f_i \cdot W_{dec}^{(i)} + b_{dec}
$$

其中 $f_i \ge 0$ 是特征激活强度，$W_{dec}^{(i)}$ 是特征的方向向量（语义含义）。

### 1.3 架构选择：Top-K SAE (SOTA 标准)

相比于传统的 ReLU SAE（难以平衡 L1 系数与稀疏度），近期论文（如 OpenAI, Anthropic 及 DeepSeek 相关研究）普遍推荐使用 **Top-K SAE**。它直接设定稀疏度 $k$，训练更稳定，特征质量更高。

### 1.3.1 前向传播 (Forward Pass)

1. **预处理**：输入 $x \in \mathbb{R}^{d_{model}}$ (通常减去 Decoder Bias 进行中心化)。
2. **编码 (Encoder)**：映射到高维并计算激活值。
    
    $$
    z = \text{ReLU}(W_{enc}(x - b_{dec}) + b_{enc})
    $$
    
3. **Top-K 激活 (Activation)**：仅保留最大的 $k$ 个值，其余置零。这是实现稀疏性的关键步骤。
    
    $$
    f = \text{TopK}(z, k)
    $$
    
    - 实现注：`values, indices = topk(z, k)`; `f = scatter(zeros, indices, values)`。
4. **解码 (Decoder)**：重构原始信号。
    
    $$
    \hat{x} = W_{dec} f + b_{dec}
    $$
    

### 1.3.2 损失函数 (Loss Function)

为了彻底解决“死神经元 (Dead Latents)”问题，采用 MSE 重建误差与辅助损失 (Auxiliary Loss) 相结合的策略。

$$
\mathcal{L} = \underbrace{||x - \hat{x}||_2^2}_{\text{MSE Reconstruction Loss}} + \lambda \cdot \underbrace{\mathcal{L}_{aux}}_{\text{Dead Latent Reconstruction}}
$$

- **Auxiliary Loss**: 针对当前 Batch 中未被激活的“死特征”，尝试用它们来预测 SAE 的重建残差 $e = x - \hat{x}$。这能为死特征提供梯度信号，使其从无效区域移动到高误差区域，从而“复活”。

---

## 第二部分：工程复现规划 (Engineering Implementation)

本部分按照 **Staged Offline (分阶段离线)** 架构划分，以解耦推理与训练，最大化显存效率。

### Phase 1: 离线激活值生成 (Offline Activation Generation)

**理论 Base**: SAE 学习的是“数据分布中的特征”。对于推理模型 (Reasoning Models)，必须包含推理过程中的思维链 (CoT)。为了避免显存瓶颈，采用“生成-保存-训练”的分离模式。

- **实现目标**: 构建一个能够流式运行模型、提取特定层激活值并保存为分片文件 (`.safetensors`) 的 Generator。
- **数据源配置 (80/20 Mix)**:
    - **推理语料 (80%)**: OpenR1-Math, GSM8K, MATH。*关键逻辑：必须包含 `<think>` 过程，以捕捉 "Society of Thought" 特征。*
    - **通用语料 (20%)**: FineWeb-Edu。*关键逻辑：用于锚定基础语言语义，防止特征空间坍塌。*
- **实现步骤**:
    1. **数据清洗与切割 (Parser & Splitter)**: 完整保留 Query 作为上下文输入以维持模型状态。在提取激活值时，**以 `\n\n` 为界限对推理过程 (CoT) 进行逻辑分段**，仅针对分段内的 Token 提取并保存激活向量，从而在保留完整语义背景的同时实现细粒度的特征对齐。
    2. **Hook 注入**: 在目标层注册 forward hook，提取 `residual_stream`。
    3. **Buffer 与分片**:
        - 运行模型 (Inference Mode, BF16)，Batch Size 设为显存允许的最大值。
        - **Global Shuffle**: 在内存中维护一个 500MB+ 的 Buffer，数据进入后打散，以满足独立同分布 (i.i.d.) 假设。
        - **Disk Write**: 当 Buffer 满时，将其写入磁盘作为分片文件 (Shard)。
- **Tensor Shape**: `[Buffer_Size, d_model]` (e.g., 100MB per file)。

### Phase 2: 模型架构实现 (Model Implementation)

**理论 Base**: 采用 Top-K SAE 架构，并利用严格的权重归一化防止解码器权重爆炸。

- **代码结构**: `class TopKSAE(nn.Module)`
- **关键组件**:
    - `W_enc`: `[d_model, d_sae]`
    - `W_dec`: `[d_sae, d_model]`
    - `b_enc`: `[d_sae]`, `b_dec`: `[d_model]`
- **初始化策略 (Initialization)**:
    - `b_dec`: 初始化为训练数据的几何中位数 (Geometric Median) 或均值。
    - `b_enc`: 初始化为 0。
    - `W_enc`, `W_dec`: Kaiming Uniform，且 $W_{dec}$ 列向量归一化。
- **关键约束 (Hard Constraint)**: **Decoder Unit Norm**。
    - 在每次 Optimizer Step 后，必须强制将 $W_{dec}$ 的每一列模长归一化为 1。这是 Top-K SAE 正常工作的物理基础。
    - *Code Snippet*: `@torch.no_grad def set_decoder_norm(self): self.W_dec.data = F.normalize(self.W_dec.data, dim=0)`

### Phase 3: 离线训练流程 (Offline Training Pipeline)

**理论 Base**: 利用磁盘上的预计算激活值进行高吞吐训练。

- **优化器**: AdamW (Beta1=0.9, Beta2=0.999, Weight Decay=0)。
- **Learning Rate**: 5e-5 (针对 1.5B/8B 模型)，配合 Warmup (前 5% steps) 和 Cosine Decay。
- **Batch Size**: **4096 ~ 8192**。离线模式下显存充足，大 Batch 有助于稀疏特征的梯度估计。
- **Auxiliary Loss**: 启用 Dead Latent Reconstruction Loss，权重 $\lambda \approx 1/32$ (经验值，需微调)。
- **验收指标 (Logging)**:
    - `L2 Ratio`: $||x - \hat{x}||^2 / ||x||^2$。目标 < 0.05。
    - `L0`: 有效稀疏度，应接近设定的 $k$。
    - `Dead Feature %`: 目标 < 5%。

### Phase 4: 评估与验证 (Evaluation & Validation)

**理论 Base**: 验证 SAE 是否学习到了真实的特征，而非噪声。

- **Step 4.1: 特征保真度 (Fidelity)**
    - **CE Loss Score**: 将 SAE 替换回原模型，计算在 Wikitext-2 和 GSM8K 上的 Perplexity 增加幅度。增加 < 5% 为合格。
- **Step 4.2: 稀疏度分析 (Sparsity)**
    - 检查 **Feature Utilization**。绘制特征激活频率的直方图，确保没有大量特征处于 "Always On" 或 "Always Off" 状态。
- **Step 4.3: 可解释性验证 (Interpretability)**
    - **Max-Activating Examples**: 对每个特征，找到 Top 20 激活的文本。
    - **CoT 语义检查**: 人工检查高激活特征是否对应特定的推理步骤（如“验算步骤”、“自我纠错”、“总结陈词”）。

### Phase 5: 结构分析与干预 (Structure & Steering)

- **Step 5.1: UMAP 可视化**
    - 提取 $W_{dec}$，运行 UMAP。观察是否有清晰的 "Islands"（如数学运算簇、逻辑连接词簇）。
- **Step 5.2: 激活共现 (Co-occurrence)**
    - 计算特征间的 Jaccard 相似度，寻找推理回路。
- **Step 5.3: 干预 (Steering)**
    - **Induction**: 找到 "Self-Correction" 特征向量，在模型回答错误时注入。

---

## 第三部分：具体实验配置 (Specific Settings)

针对您提出的目标模型，采用 **Staged Offline** 策略的参数配置如下。

### 3.1 模型参数概览

| **配置项** | **DeepSeek-R1-Distill-Qwen-1.5B** | **Qwen3-8B** | **QwQ-32B** |
| --- | --- | --- | --- |
| **d_model** | 1,536 | 4,096 | 5,120 |
| **Layer to Hook** | Layer 12 | Layer 16 | Layer 24 |

### 3.2 SAE 训练参数 (基于 Top-K + Offline)

| **参数** | **推荐值** | **说明** |
| --- | --- | --- |
| **Expansion Factor** | **32x** | 推理模型需要更高的膨胀系数以分离细粒度逻辑特征。 |
| **$d_{sae}$ (1.5B)** | **49,152** | 32 * 1536. |
| **$d_{sae}$ (8B)** | ~131,072 | 32 * 4096. |
| **Top-K ($k$)** | 32 ~ 64 | 1.5B 建议 32，8B 建议 64。 |
| **Batch Size** | **8192** | 离线训练显存无压力，大 Batch 更有利于收敛。 |
| **Context** | Filtered CoT | 仅保留思维链片段。 |
| **Learning Rate** | 5e-5 | 配合 Warmup。 |
| **Total Tokens** | **400M** | 确保覆盖长尾稀疏特征。 |

### 3.3 硬件需求估算 (离线模式)

- **Stage 1 (Generation)**: 单卡 4090/A100 即可。瓶颈在于磁盘 I/O。
- **Stage 2 (Training)**: 单卡 4090 (24GB) 即可训练 1.5B 规模的 SAE (Batch 8192)。
- **存储**: 需准备约 **1TB** 高速 NVMe SSD 存储激活值分片 (滚动覆盖模式可减少至 200GB)。

---

## 第四部分：执行 Checklist (Action Plan)

请按以下顺序执行复现：

1. **环境搭建**:
    - 安装 `transformer_lens`, `flash_attn`, `safetensors`.
2. **Phase 1 (Data Gen)**:
    - 编写 `Module A` (MixedTokenSource) 和 `Module B` (ActivationGenerator)。
    - 启动生成脚本，检查生成的 `.safetensors` 文件大小和内容分布。
3. **Phase 2 (Arch)**:
    - 实现 `TopKSAE` 类，包含 `Auxiliary Loss` 和 `set_decoder_norm`。
    - 编写 `unittest` 验证归一化约束是否生效。
4. **Phase 3 (Train - 1.5B)**:
    - 启动离线训练器 `Module D`。
    - 监控 `L2 Ratio` 和 `Dead Feature %`。
5. **Phase 4 (Analysis)**:
    - 抽取 Top 10 活跃特征，人工验证其对应的 CoT 文本片段。
    - 计算 CE Loss Score，确保模型智力未受损。
6. **Scale Up**:
    - 确认 1.5B 流程跑通后，扩展至 8B/32B 模型。