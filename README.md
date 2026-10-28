# liquiforge

> LFM3 风格 **tri-hybrid 全模态统一骨干**零依赖参考实现 —— 一个骨干，五种模态
> （文本 / 视觉 / 音频 / omni / 决策），纯 Python、零第三方依赖。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Zero Deps](https://img.shields.io/badge/dependencies-0-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![CI](https://img.shields.io/badge/CI-unittest-brightgreen)

**English** — A zero-dependency reference implementation of an LFM3-style
*tri-hybrid* omni-modal backbone: **linear attention + depthwise convolution +
sparse Mixture-of-Experts** in one block, shared across text / vision / audio /
omni / decision. One base backbone is fine-tuned into five task variants using
lightweight heads + LoRA.

---

## 这个项目拆解了什么技术

Liquid AI 的 LFM3 用「一个骨干吃下文本、视觉、音频、全模态与决策」——关键不在
榜单，而在**架构**：怎么让一种网络同时具备全局、局部、高容量三种归纳偏置，又
怎么从一个 Base 低成本微调出多款模型。本仓库把它拆成可运行、可证伪的代码。

### 1. tri-hybrid block：三种算子的「并联 + 门控」

```
block(x) = x + gate_a·LinearAttention(x) + gate_c·DepthwiseConv(x) + gate_m·MoE(x)
```

| 算子 | 解决什么 | 复杂度 | 关键技巧 |
|------|----------|--------|----------|
| **线性注意力** | 全局依赖 | O(n·d²)，不随序列平方 | 正特征映射 φ(x)=ELU(x)+1，递推 `S←S+φ(k)vᵀ`，**必须同时累加归一化 z** |
| **深度卷积** | 局部结构 | O(n·k·d) | 因果、每通道独立 kernel，只回看过去 |
| **稀疏 MoE** | 容量 | 每 token 只跑 top-k | top-k 路由 + **负载均衡辅助损失** + 路由噪声防抖 |

三者并联而非串联：不同任务对不同归纳偏置的依赖不同，门控让网络自己学配比。

### 2. 统一 token 空间

文本 BPE、图像 patch、音频 STFT→mel→VQ 三种编码器，全部映射到**同一张嵌入表**的
相邻区段（`(modality, local_id)` → 全局 id）。共享嵌入是跨模态对齐的物理基础——
这是「一个骨干吃全模态」能成立的前提。

### 3. 一个 Base 出五款模型

共享主干 + 轻量任务头 + **LoRA 低秩适配器**：

- 文本头 = 因果 LM（next-token）；视觉头 = 分类；音频头 = 转写；决策头 = 策略；
- omni 头 = 共享投影空间的跨模态检索。
- 微调时**冻结主干**，只训 `A@B`（rank≈8）残差，`scale = alpha/rank`，增量参数
  只占全量零头。

### 4. 决策模态 = 策略梯度 RL

决策头用 REINFORCE 风格策略梯度，配一个**确定性奖励 oracle**（正确答案唯一），
这样成功率曲线涨跌可归因、可证伪。

---

## 快速开始（零依赖）

```bash
# 冒烟：五个头在共享骨干上各跑一遍
python -m liquiforge.cli doctor

# 全量基准：skill 0.5(随机基线) -> ~1.0(oracle 主导) 的单调变化
python -m liquiforge.cli bench

# 确定性训练
python -m liquiforge.cli train --target 0.95

# stdlib HTTP 服务（GET /health /v1/models /v1/metrics /v1/bench）
python -m liquiforge.cli serve --port 8787
```

安装（可选）：`pip install -e .`

---

## 基准（真实测量，skill 0.5 → 1.0）

| 指标 | 基线(0.5) | 训练后(~1.0) | 方向 |
|------|-----------|--------------|------|
| text perplexity | 282.35 | 2.49 | ↓ 越低越好 |
| vision accuracy | 0.25 | 1.00 | ↑ |
| audio WER | 5.50 | 0.00 | ↓ 越低越好 |
| omni recall@1 | 0.50 | 1.00 | ↑ |
| decision success | 0.00 | 1.00 | ↑ |

运行 `python scripts/run_bench.py` 复现。

---

## 目录结构

```
liquiforge/
  backbone/    tri-hybrid 骨干（线性注意力/深度卷积/MoE/block/model）
  heads/       五个任务头（text/vision/audio/omni/decision）
  adapters/    LoRA 低秩适配器
  rl/          策略梯度 + 确定性奖励
  tokenizers/  统一 token 化（BPE/图像 patch/音频 VQ）
  eval/        各模态指标
  train.py     确定性 skill 驱动训练
  bench.py     基准
  serving/     stdlib HTTP
examples/       7 个可运行示例
tests/          unittest 单测
docs/           架构 / 指南 / 可复用要点 / 提示词
prompts/        提示词模板（可直接改）
recipes/        YAML 配置配方（tiny/default/large/decision_rl）
```

## 可复用要点（TL;DR）

1. **线性注意力必加归一化累加器 z**，特征映射必须为正（ELU+1），否则数值发散。
2. **局部卷积是对线性注意力的必要补丁**——相邻 token 强相关的任务必加。
3. **MoE 必须带负载均衡辅助损失**，路由加噪声防退化成固定专家。
4. **一个骨干出多款 = 共享嵌入 + 共享主干 + 轻头 + LoRA**，四件套缺一不可。
5. **决策 RL 先定义确定性 reward oracle**，否则成功率涨跌无法归因。

详见 `docs/reusable_takeaways.md` 与 `docs/prompting.md`（含可直接复用的提示词模板）。

## License

MIT
