# 使用指南

## 安装（零依赖）

```bash
pip install -e .        # 或直接 python -m liquiforge.cli ...
```

## 快速上手

```bash
# 冒烟：五个头在共享骨干上各跑一遍
python -m liquiforge.cli doctor

# 全量基准（skill 0.5 -> ~1.0 的单调变化）
python -m liquiforge.cli bench

# 确定性训练
python -m liquiforge.cli train --target 0.95

# 启动 stdlib HTTP 服务
python -m liquiforge.cli serve --port 8787
```

HTTP 端点：`GET /health`、`/v1/models`、`/v1/metrics`、`/v1/bench`。

## 目录

```
liquiforge/
  backbone/    tri-hybrid 骨干（注意力/卷积/MoE/block/model）
  heads/       五个任务头（text/vision/audio/omni/decision）
  adapters/    LoRA 低秩适配器
  rl/          决策策略梯度 + 奖励
  tokenizers/  统一 token 化（BPE/图像 patch/音频 VQ）
  eval/        各模态指标
  train.py     确定性 skill 驱动训练
  bench.py     基准
  serving/     stdlib HTTP
examples/       7 个可运行示例
tests/          单元测试（unittest 发现）
docs/           架构与指南
prompts/        提示词模板
recipes/        YAML 配置配方
```

## 配置

`config.yaml` 或 recipes 目录下的 YAML 覆盖默认值（model/training/serving 三块）。
