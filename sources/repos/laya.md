# Laya（NandhaKishorM/laya · Convai Innovations）

- **标题:** Laya
- **链接:** https://github.com/NandhaKishorM/laya
- **类型:** repo / decision-engine / system-one
- **作者:** Nandha Kishor M（Convai Innovations）
- **许可:** Apache-2.0
- **PyPI:** https://pypi.org/project/laya/
- **Hugging Face 权重:** https://huggingface.co/convaiinnovations/laya（含 `multilingual`、`typed-decisions` 子目录）
- **HF Demo:** https://huggingface.co/spaces/convaiinnovations/laya-demo
- **MLX 端口（社区，Apple Silicon）:** https://github.com/mizorewww/laya-mlx · HF 预转换：[laya-mlx](https://huggingface.co/aac6fef/laya-mlx)、[multilingual](https://huggingface.co/aac6fef/laya-multilingual-mlx)、[typed-decisions](https://huggingface.co/aac6fef/laya-typed-decisions-mlx)
- **Core ML（Swift，FluidInference）:** [FluidInference/laya-coreml](https://huggingface.co/FluidInference/laya-coreml) · 消费方 [FluidUse](./fluiduse.md)
- **Core ML（Python，社区）:** https://github.com/mizorewww/laya-coreml · PyPI `laya-coreml` · 展示页 [madewithlaya-laya-coreml](../sites/madewithlaya-laya-coreml.md)
- **Colab:** https://colab.research.google.com/drive/15d4Yv__KHeHjshVb-6PRTfqVllxih2S3
- **工程文章:** https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me
- **最后核查:** 2026-09-20
- **入库日期:** 2026-09-20

## 开源状态（步骤 2.5）

- **已开源：** GitHub 推理/训练代码 + Apache 2.0 权重（HF 三 checkpoint）+ `pip install laya`；可自托管，无 API 密钥依赖。

## 核心内容摘要

1. **非自回归 System 1 决策引擎：** 对结构化 state（文本/邮件/JSON）在**单次前向**回答 typed questions（`choice` / `score` / `noul`），不生成自由文本。
2. **三 checkpoint + Router：** `laya`（英文 ModernBERT-large）、`laya-multilingual`（100+ 语言 mmBERT）、`laya-typed-decisions`（工作流微调）；`Router(preload=True)` 按脚本/语言自动选模型（<0.5 ms 纯 Python 检测）。
3. **RLCD 训练：** strictly proper scoring rules 的强化学习，置信度可用于自动化门控（官方 ECE 经 temperature 拟合后 ~0.081）。
4. **性能（T4 实测）：** 单问 ~33 ms；10 问 batch ~7.2 ms/问；相对 TypeSafe Jev 公开 benchmark 约 6–8× 更快（第三方数字，见 README `BENCHMARKS.md`）。
5. **诚实边界：** base checkpoint 在 typed-decisions **零样本接近随机**（~0.36）；主要价值在领域微调（微调后 0.766 vs Jev 0.727）；>50 选项 choice 需调 `head_max_len` 或分层 choice。

## 对 wiki 的映射

- **wiki/entities/laya.md** — 框架实体（与 [typesafe-jev](../../wiki/entities/typesafe-jev.md) 对照）
- **wiki/entities/laya-mlx.md** — Apple Silicon MLX 推理端口（[laya-mlx.md](./laya-mlx.md)）
- **wiki/entities/laya-coreml.md** · **wiki/entities/fluiduse.md** — Core ML 双栈与 Swift computer-use（[laya-coreml-fluidinference.md](./laya-coreml-fluidinference.md)、[fluiduse.md](./fluiduse.md)）
- **wiki/concepts/behavior-tree-vla-orchestration.md** — 毫秒级路由/guardrail 与 VLA 分层
- **wiki/concepts/llm-robotics-control-interfaces.md** — LLM 接口抽象阶梯上的「结构化决策」层
