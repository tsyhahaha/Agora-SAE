# 实验文档：MATH500 论文复现主线

**实验代号**: `Exp-01-MATH500-1.5B`

**文档目标**:
- 给出一条当前仓库内可逐步执行的 `MATH500` 复现路径。
- 优先对齐论文主线: `step-level behavior labeling -> final-layer geometry -> causal intervention`。
- 同时把 `layer-wise` 扫描所需的激活生成和 SAE 训练路径沉淀下来，为后续跨层 silhouette 分析做准备。

---

## 1. 先说结论

如果你的目标是“先达到论文复现的一阶段核心结果”，推荐按下面顺序跑:

1. 采样固定 `MATH500`
2. 生成 `final layer` 的 step-delimiter activations
3. 训练 `final layer SAE`
4. 做 step 行为标注
5. 做 `final layer` 的 geometry 分析
6. 做 `final layer` intervention

如果你还要补论文里的 `layer-wise silhouette` 分析，再额外做:

7. 按固定步长生成多层 activation
8. 按相同步长训练多层 SAE
9. 逐层跑 geometry，并汇总各层 `silhouette_score`

一句话概括:
- `final layer` 主线是现在最该优先跑通的复现路径。
- `layer-wise scan` 是主线之外的必要扩展，但当前仓库还是“一层一层跑 geometry”，还没有自动化的多层 eval 汇总 CLI。

---

## 2. 当前仓库与论文目标的关系

### 2.1 当前仓库已经对齐的部分

- 激活提取已经改成 `step delimiter`，不再保存整段 `token span`。
- 有限本地数据集默认单轮结束，不会无限循环生成 activation。
- `paper-style eval` 已经有专门入口 [evaluate_paper_math500.py](/Users/siyuantao/repos/Agora-SAE/agora_sae/scripts/evaluate_paper_math500.py)，覆盖:
  - `label-steps`
  - `analyze-geometry`
  - `run-intervention`
- 训练侧已经支持多层扫描:
  - `--layers`
  - `--layer-range`
  - `--layer-step`
  - `--final-layer`

### 2.2 当前仓库和论文仍然存在的差异

- 论文 `Section 4.2` 用的是**标准 SAE**，当前仓库训练入口还是 **Top-K SAE**。
- 论文正文的核心图和 intervention 主结果聚焦 `final layer`；当前仓库的 `math500-1.5b` preset 默认层仍是 `12`，所以走论文主线时需要显式传 `--layer 27`。
- 论文里 layer-wise 几何量化是统一汇总出来的；当前仓库要先逐层生成 `geometry_summary.json`，再手动汇总。

所以这份文档的定位是:
- 先把**论文复现主线**在当前代码库里跑通。
- 明确保留当前仓库的一个剩余偏差: `Top-K SAE != 标准 SAE`。

---

## 3. 实验固定输入

### 3.1 模型与数据

- **Target Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Local Model Path**: `<LOCAL_MODEL_PATH>`
- **Source Dataset Path**: `<LOCAL_DATASET_PATH>`
- **Sampled Dataset Path**: `<LOCAL_MATH500_PATH>`

### 3.2 Step 切分与 activation 取点

当前复现路径固定采用:

- `step delimiter = "\n\n"`
- 对 reasoning 按 `\n\n` 切 step
- 每个 step 只取一个 activation point:
  - 优先取该 step 对应的 delimiter token
  - 如果最后一个 step 后没有显式 delimiter，则回退到该 step 末 token

这条策略的目的，是把:
- “怎么切 step”
- “在哪个点取 activation”

从代码结构上解耦开。后续如果你要换规则式切分、句子级切分、或 LLM judge 给出的 step anchor，activation 主流程不用重写。

### 3.3 论文主线建议层位

对 `DeepSeek-R1-Distill-Qwen-1.5B`，当前仓库里我们把:

- `final layer index = 27`

作为论文主线默认层位。

因此:
- `final layer` 主线命令都建议显式写 `--layer 27`
- `layer-wise scan` 的示例层集合统一使用:
  - `0, 4, 8, 12, 16, 20, 24, 27`

这样既能按固定步长抽层，也能保证 `final layer` 一定包含进去。

---

## 4. Phase 0: 生成固定的 MATH500 子集

先从本地数学数据集中采样固定 `500` 条，后续所有实验都基于这个固定子集。

```bash
python -m agora_sae.scripts.sample_dataset \
    --dataset-path <LOCAL_DATASET_PATH> \
    --output-path <LOCAL_MATH500_PATH> \
    --num-samples 500 \
    --seed 42
```

建议先确认输出目录里已经是一个可被 Hugging Face `datasets` 正常读取的本地数据集目录，然后再进入下一阶段。

---

## 5. Phase 1: 生成 activations

这一阶段分成两条路径:

- `Phase 1A`: 论文主线必须跑的 `final layer activation`
- `Phase 1B`: 为 layer-wise scan 准备的多层 activation

### Phase 1A: final layer activation

这是复现主线的最低要求。先只生成 `layer 27` 的 activations。

```bash
python -m agora_sae.scripts.generate_activations \
    --preset math500-1.5b \
    --model <LOCAL_MODEL_PATH> \
    --layer 27 \
    --reasoning-datasets <LOCAL_MATH500_PATH> \
    --output ./data/math500_activations/layer_27 \
    --batch-size 16 \
    --buffer-size-mb 500 \
    --shard-size-mb 100
```

这条命令的关键点:
- 虽然用了 `math500-1.5b` preset，但这里显式传了 `--layer 27`，所以会覆盖 preset 默认的 `layer 12`。
- 对本地有限数据集，脚本默认单轮结束，不需要额外传 `--max-batches`。
- 输出目录建议直接按 `layer_x` 组织，方便后面训练侧和 layer-wise 扫描复用。

**验收点**:
- `./data/math500_activations/layer_27` 下出现 `.safetensors` 分片
- 生成过程会自然结束，而不是无限循环

### Phase 1B: 可选的多层 activation 扫描

如果你要复现论文里的 layer-wise 几何量化，需要先为多个层分别生成 activations。

当前仓库的 activation 生成还是**单层一次跑一个 layer**，所以这里建议直接用一个 shell loop。

```bash
for layer in 0 4 8 12 16 20 24 27; do
  python -m agora_sae.scripts.generate_activations \
      --preset math500-1.5b \
      --model <LOCAL_MODEL_PATH> \
      --layer ${layer} \
      --reasoning-datasets <LOCAL_MATH500_PATH> \
      --output ./data/math500_activations/layer_${layer} \
      --batch-size 16 \
      --buffer-size-mb 500 \
      --shard-size-mb 100
done
```

**为什么这里必须先做 Phase 1B 再做多层训练**:
- 多层 SAE 训练依赖“每一层都有自己的 activation shards”
- 当前训练侧已经支持多层 sweep，但不会替你自动生成多层 activation

**验收点**:
- `./data/math500_activations/layer_0`
- `./data/math500_activations/layer_4`
- ...
- `./data/math500_activations/layer_27`

这些目录都已经存在，并各自包含 activation 分片。

---

## 6. Phase 2: 训练 SAE

这一阶段同样分成两条路径:

- `Phase 2A`: 先训练 `final layer SAE`
- `Phase 2B`: 再训练多层 SAE，为跨层几何分析做准备

### Phase 2A: 训练 final layer SAE

```bash
python -m agora_sae.scripts.train_sae \
    --preset math500-1.5b \
    --layer 27 \
    --shards ./data/math500_activations/layer_27 \
    --checkpoint-dir ./checkpoints/math500-final-layer \
    --batch-size 4096 \
    --lr 5e-5 \
    --steps 100000 \
    --no-wandb
```

这条命令是当前仓库里最接近论文主线的训练入口。

**关键说明**:
- 这里继续显式传 `--layer 27`，确保训练配置和 activation 来源一致。
- 当前实现仍然训练的是 `Top-K SAE`，不是论文中的标准 SAE；这是当前仓库里还没有消除的一处剩余偏差。

**验收点**:
- `./checkpoints/math500-final-layer/checkpoint_final.pt` 成功产出

### Phase 2B: 可选的多层 SAE 扫描训练

如果你已经完成 `Phase 1B` 的多层 activation 生成，就可以直接让训练脚本按层扫描。

```bash
python -m agora_sae.scripts.train_sae \
    --preset math500-1.5b \
    --layer-range 0:24 \
    --layer-step 4 \
    --final-layer 27 \
    --shards ./data/math500_activations \
    --checkpoint-dir ./checkpoints/math500-layer-scan \
    --batch-size 4096 \
    --lr 5e-5 \
    --steps 100000 \
    --no-wandb
```

这条命令会顺序训练:

- `0`
- `4`
- `8`
- `12`
- `16`
- `20`
- `24`
- `27`

其中 `27` 是自动补上的 `final layer`。

脚本会默认把输入 shards 解析成:
- `./data/math500_activations/layer_0`
- `./data/math500_activations/layer_4`
- ...

输出 checkpoint 解析成:
- `./checkpoints/math500-layer-scan/layer_0`
- `./checkpoints/math500-layer-scan/layer_4`
- ...

并自动写出:
- `./checkpoints/math500-layer-scan/scan_manifest.json`

这份 manifest 后面可以作为 layer-wise eval 的统一索引。

**注意**:
- 如果你只跑了 `Phase 1A` 的 `layer_27` activation，就不要直接执行这条命令。
- 多层训练之前，先确认多层 activation 目录都已经存在。

---

## 7. Phase 3: 论文主线 eval

这里的评估目标不是通用重建误差或 PPL，而是论文最核心的三步:

1. `Behavior labeling`
2. `Decoder geometry`
3. `Causal intervention`

### Step 3.1: 做 step 行为标注

先让目标模型对 `MATH500` 生成 response，再按 `\n\n` 切 step，最后用外部 judge 给 step 打标。

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

如果中途因为网络抖动或 judge API 失败中断，直接用同一条命令加上 `--resume` 即可继续，已经写入 `JSONL` 的 step 会被自动跳过：

```bash
python -m agora_sae.scripts.evaluate_paper_math500 label-steps \
    --dataset-path <LOCAL_MATH500_PATH> \
    --output ./eval/math500_step_labels.jsonl \
    --response-source model \
    --model <LOCAL_MODEL_PATH> \
    --judge minimax \
    --judge-model MiniMax-M2.5 \
    --max-samples 500 \
    --max-new-tokens 512 \
    --resume
```

如果你明确要丢弃旧结果重新开始，再加 `--overwrite-output`。

如果你要改用 MiniMax 做 labeling，可以切到下面这条命令：

```bash
export MINIMAX_API_KEY=<YOUR_MINIMAX_API_KEY>
# 中国区可按需覆盖:
# export MINIMAX_BASE_URL=https://api.minimaxi.com/v1

python -m agora_sae.scripts.evaluate_paper_math500 label-steps \
    --dataset-path <LOCAL_MATH500_PATH> \
    --output ./eval/math500_step_labels.jsonl \
    --response-source model \
    --model <LOCAL_MODEL_PATH> \
    --judge minimax \
    --judge-model MiniMax-M2.5 \
    --max-samples 500 \
    --max-new-tokens 512
```

**产物**:
- `./eval/math500_step_labels.jsonl`

**建议检查**:
- 抽样看几条记录，确认存在:
  - `sample_id`
  - `step_id`
  - `step_text`
  - `label`

如果你只是要先跑通本地流程，可以把 `--judge openai` 或 `--judge minimax` 改成 `--judge heuristic`，但那不属于论文主线。

### Step 3.2: 做 final-layer geometry 分析

这是论文主线里最先该跑通的 geometry 结果。

```bash
python -m agora_sae.scripts.evaluate_paper_math500 analyze-geometry \
    --labels ./eval/math500_step_labels.jsonl \
    --checkpoint ./checkpoints/math500-final-layer/checkpoint_final.pt \
    --model <LOCAL_MODEL_PATH> \
    --layer 27 \
    --output-dir ./eval/geometry_math500_final \
    --embedding-method umap \
    --plot-path ./eval/geometry_math500_final/decoder_umap.png
```

**验收点**:
- `./eval/geometry_math500_final/geometry_summary.json`
- `./eval/geometry_math500_final/decoder_points.jsonl`
- `./eval/geometry_math500_final/decoder_umap.png`

其中 `geometry_summary.json` 至少应包含:
- `silhouette_score`
- `feature_assignments`
- `num_labeled_steps`
- `num_captured_steps`

### Step 3.3: 做 final-layer intervention

拿 `geometry_summary.json` 里的 behavior-specific decoder columns，构造行为向量，再注入回原模型。

下面先以 `reflection` 为例:

```bash
python -m agora_sae.scripts.evaluate_paper_math500 run-intervention \
    --dataset-path <LOCAL_MATH500_PATH> \
    --geometry-summary ./eval/geometry_math500_final/geometry_summary.json \
    --checkpoint ./checkpoints/math500-final-layer/checkpoint_final.pt \
    --model <LOCAL_MODEL_PATH> \
    --layer 27 \
    --behavior reflection \
    --output ./eval/intervention_reflection.jsonl \
    --judge openai \
    --judge-model gpt-5 \
    --conditions negative:1.0,vanilla:0.0,positive:-1.0 \
    --max-samples 32 \
    --max-new-tokens 384
```

这一步同样支持 `--resume`。如果中途断掉，再次执行相同命令并加 `--resume`，已经完成的 `sample_id + condition` 组合会被跳过，不会从头再跑。

如果你要用 MiniMax 作为行为计数 judge，把上面命令里的
`--judge openai --judge-model gpt-5`
替换成
`--judge minimax --judge-model MiniMax-M2.5`
即可。中国区如果需要，也可以先设置 `MINIMAX_BASE_URL=https://api.minimaxi.com/v1`。

如果你要做 `backtracking`，只需要把:
- `--behavior reflection`

改成:
- `--behavior backtracking`

**验收点**:
- 输出 JSONL 里，每个条件下至少能看到:
  - 行为步数统计
  - 最终答案
  - 正确性字段

---

## 8. Phase 4: layer-wise geometry 扫描

这一阶段对应论文里“不同层行为可分离性”的量化分析。

**当前仓库现状**:
- 训练侧已经支持多层扫描
- eval 侧还没有“一条命令直接扫完所有层”的 CLI
- 所以当前推荐做法是: **逐层跑 `analyze-geometry`，再汇总 `silhouette_score`**

### Step 4.1: 逐层跑 geometry

前提:
- 你已经完成 `Phase 1B`
- 你已经完成 `Phase 2B`
- `./checkpoints/math500-layer-scan/scan_manifest.json` 已经存在

```bash
for layer in 0 4 8 12 16 20 24 27; do
  python -m agora_sae.scripts.evaluate_paper_math500 analyze-geometry \
      --labels ./eval/math500_step_labels.jsonl \
      --checkpoint ./checkpoints/math500-layer-scan/layer_${layer}/checkpoint_final.pt \
      --model <LOCAL_MODEL_PATH> \
      --layer ${layer} \
      --output-dir ./eval/layer_scan/layer_${layer} \
      --embedding-method pca
done
```

这里推荐先用 `pca`，因为:
- layer-wise 扫描的核心是比较 `silhouette_score`
- `pca` 更轻，排查环境问题也更简单
- 等你确认主要趋势后，再把重点层切回 `umap`

### Step 4.2: 汇总各层 silhouette 分数

当前可以直接用一个标准库 Python 脚本做最小汇总:

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("./eval/layer_scan")
rows = []
for path in sorted(base.glob("layer_*/geometry_summary.json")):
    layer = int(path.parent.name.split("_")[-1])
    data = json.loads(path.read_text())
    rows.append((layer, data.get("silhouette_score")))

print("layer\tsilhouette_score")
for layer, score in rows:
    print(f"{layer}\t{score}")
PY
```

这一步的目标是回答:
- later layers 是否比 earlier layers 更容易把 `reflection / backtracking / other` 区分开
- `final layer` 是否仍然是主实验层的合理选择

**推荐解释方式**:
- 如果 `final layer` 的 silhouette 已经接近最高，后续主结果继续以 `final layer` 为中心
- 如果某个更早层更高，也建议把它记录为补充观察，而不是直接替换 `final layer` 主线

---

## 9. 最终推荐的运行顺序

如果你现在就是要一步步跑到“论文复现第一阶段”的目标，推荐严格按下面顺序执行:

1. `Phase 0`: 采样 `MATH500`
2. `Phase 1A`: 生成 `layer_27` activations
3. `Phase 2A`: 训练 `layer_27 SAE`
4. `Step 3.1`: 做 step label
5. `Step 3.2`: 做 final-layer geometry
6. `Step 3.3`: 做 final-layer intervention

如果上面都跑通，再继续:

7. `Phase 1B`: 生成多层 activations
8. `Phase 2B`: 训练多层 SAE
9. `Step 4.1`: 逐层 geometry
10. `Step 4.2`: 汇总 silhouette 分数

这样做的好处是:
- 先拿到论文主线最关键的 final-layer 结果
- 再补 layer-wise 分析
- 不会把“主线复现”和“扫描扩展”混在一起，导致阶段之间互相卡住

---

## 10. 资源预估

### 10.1 final-layer 主线

- **Activation generation**:
  - 1.5B 模型 bf16 推理大致需要 `~12GB` 量级显存
  - `batch-size=16` 通常是较稳妥的起点
- **SAE training**:
  - 当前 `Top-K SAE` 训练显存压力较小，通常 `<4GB`
- **Disk**:
  - step-delimiter activation 只保存 step-level 向量
  - `500` 条数学题通常只落到 `数十 MB 到数百 MB`

### 10.2 layer-wise 扫描

- Activation generation 和 SAE training 的总耗时，会近似按层数线性放大
- 如果扫描 `0,4,8,12,16,20,24,27` 这 `8` 层，整体成本大约是 final-layer 单层实验的 `8x`
- 所以推荐先跑通 `final layer` 主线，再决定是否补全多层扫描

---

## 11. 最低验收标准

### 11.1 主线验收

1. `./data/math500_activations/layer_27` 成功生成 activation shards
2. `./checkpoints/math500-final-layer/checkpoint_final.pt` 成功产出
3. `./eval/math500_step_labels.jsonl` 成功产出
4. `./eval/geometry_math500_final/geometry_summary.json` 成功产出
5. `./eval/intervention_reflection.jsonl` 成功产出

### 11.2 结果验收

1. Geometry 结果里 `reflection` 和 `backtracking` 的 decoder columns 不是完全混杂
2. `silhouette_score` 是可计算的，不是空值
3. `intervention` 后相关行为步数能够随条件变化
4. 最终答案不会在所有样本上全面崩坏

### 11.3 layer-wise 扫描验收

1. 多层 checkpoint 目录和 `scan_manifest.json` 成功产出
2. 每个扫描层都能得到一个 `geometry_summary.json`
3. 能汇总出一张 `layer -> silhouette_score` 对照表

---

## 12. 当前脚本的定位

- 论文主线 eval 请使用 [evaluate_paper_math500.py](/Users/siyuantao/repos/Agora-SAE/agora_sae/scripts/evaluate_paper_math500.py)
- 通用诊断脚本 [evaluate_sae.py](/Users/siyuantao/repos/Agora-SAE/agora_sae/scripts/evaluate_sae.py) 可以保留，但不要再把它当成 `MATH500` 论文复现主路径

如果后续你要继续往“更严格的 faithful reproduction”推进，下一步最重要的工作不是再补文档，而是把训练侧从 `Top-K SAE` 切到论文中的**标准 SAE**。
