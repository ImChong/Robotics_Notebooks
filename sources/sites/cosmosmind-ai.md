# CosmosMind 官网

> 来源归档

- **标题：** CosmosMind — Finite Cosmos Infinite Mind
- **类型：** site / lab
- **链接：** <https://cosmosmind.ai/>
- **入库日期：** 2026-09-14
- **一句话说明：** CosmosMind AI Lab 官方入口：以 Darwin 式 meta-engine 叙事组织 MetaRSI、SWE 评测与 agent harness 研究；MetaRSI-v1 链到 PDF / GitHub / Hugging Face。
- **沉淀到 wiki：** [`wiki/entities/paper-metarsi-v1.md`](../../wiki/entities/paper-metarsi-v1.md)、[`wiki/entities/rsi-harness.md`](../../wiki/entities/rsi-harness.md)

## 开源状态（步骤 2.5）

- **MetaRSI-v1 项目页：** <https://www.cosmosmind.ai/research/metarsi-v1> — 页眉链 **PDF**、**GitHub**（[CosmosMind-ai/RSI-Harness](https://github.com/CosmosMind-ai/RSI-Harness)）、**Hugging Face**（[CosmosMind/RSI-Harness](https://huggingface.co/CosmosMind/RSI-Harness)）。
- **结论：** **已开源（Harness-RSI 实现）** — 官方 harness 仓可 clone + `./install.sh` 运行；论文侧 Model-RSI / Data-RSI 训练与 benchmark 代码 **不在** RSI-Harness 仓（HF README 明确无 benchmark / training / eval 代码）。
- **同站其它条目（本次未 ingest）：** SWE-Prometheus、SWE-PolyVision 各有独立 PDF / GitHub / HF / Homepage 链。

## 对本库的意义

- 把 **Harness-RSI** 从 MetaRSI 论文抽象落到可安装 artifact（RSIH + Genome + GEE），与 [SoL-Pi](../../wiki/entities/sol-pi.md)、[DeepSeek Harness](../../wiki/entities/deepseek-harness.md) 同属 **coding agent harness** 选型轴。
- MetaRSI 把 RSI 从「单表面 edit」扩到 **Data / Harness / Model 三算子可组合调度**，补充 [递归自改进](../../wiki/concepts/recursive-self-improvement.md) 概念页的 **中间态 harness 路线** 实例。
