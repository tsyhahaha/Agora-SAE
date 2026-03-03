# 实验文档：MATH500 上的 SAE 训练 (Experiment 1)

**实验代号**：`Exp-01-MATH500-1.5B`
**目标**：在小规模数学推理数据集 (MATH500) 上走通 SAE (Sparse Autoencoder) 的离线提取与训练全流程，验证特定模型下的数据处理切割逻辑与特征提取有效性。

---

## 1. 实验核心配置

### 1.1 模型与数据集路径
- **Target Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - **Local Path**: `/home/taosiyuan/huggingface/DeepSeek-R1-Distill-Qwen-1.5B`
- **Source Dataset**: `MATH`
  - **Local Path**: `/home/taosiyuan/huggingface/competition_math`

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

## 2. 实验执行流程

本实验遵循项目中定义的 **Staged Offline** 架构。

### Phase 1: 离线激活值生成 (Activation Generation)
1. **脚本执行**: 运行激活获取脚本，加载 `DeepSeek-R1-Distill-Qwen-1.5B` (`/home/taosiyuan/huggingface/DeepSeek...`)。
2. **数据处理**: 加载 `/home/taosiyuan/.../competition_math`，采样 500 条，执行 parser ，按 `\n\n` 截断。
3. **前向与保存**: 将截短后的片段输入模型，提取指定层激活并写入 `buffer`，满载后 shuffle 落地为 `.safetensors` 分片文件。

### Phase 2: 高吞吐训练 (Offline SAE Training)
1. **启动训练器**: 加载保存的激活特征大文件。
2. **联合优化**: 使用 MSE 重建损失，辅以 Aux Loss (死特征重激活损失)。并严格执行 Decoder 的单位范数 (Unit Norm) 约束。

### Phase 3: 特征解释与验证 (Evaluation)
抽取特征字典中的 Top-10 / Top-20 最频发活跃特征，还原到文本切片中观察。预期能捕捉到数学步骤特定的语义概念，例如：
- 特征 A: 专门在 `\n\n` 切断后的独立条件陈述句激活 (e.g., "Let $x$ be...")。
- 特征 B: 在推断转折与纠错句激活 (e.g., "Wait, this is incorrect...").
- 特征 C: 在最终计算化简步骤激活。

---

## 3. 预期结果 (Acceptance Criteria)
1. **成功产出 MATH500 激活文件**: 在磁盘上生成有效的 `.safetensors` 分片，且大小与 `MATH500` 的片段数量（Token 规模）相匹配。
2. **训练收敛**: L2 重构误差下降至预期水平 (`< 5%`)，L0 激活特征平均数维持在 `~32` 左右。
3. **0 死神经元 (Dead Latents)**: 通过 Aux Loss 确保在 MATH500 特定领域内的特征死亡率处于极低水平 (`< 1%`)。
