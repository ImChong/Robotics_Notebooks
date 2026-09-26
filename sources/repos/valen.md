# Valen（Liuziyu77/Valen · 万澜）

- **标题:** Valen — Multimodal System One Decision Model
- **链接:** https://github.com/Liuziyu77/Valen
- **类型:** repo / system-one / multimodal-decision
- **维护:** Valen Team（GitHub `Liuziyu77`；HF [Valen-Team](https://huggingface.co/Valen-Team)）
- **许可:** Apache-2.0（代码）；Qwen3.5 与源数据集沿用各自许可
- **Preview 权重:** https://huggingface.co/Valen-Team/Valen-Preview-0923
- **Base 模型:** https://huggingface.co/Qwen/Qwen3.5-2B（revision `15852e8c16360a2fea060d615a32b45270f8a8fc`）
- **在线 Demo:** https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923
- **训练集 General 100k:** https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k
- **评测集 General 5k:** https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k
- **Sokoban 训练/评测:** https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game
- **设计参照:** [TypeSafe Jev / System One Models](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- **最后核查:** 2026-09-26
- **入库日期:** 2026-09-26

## 开源状态（步骤 2.5）

| 项 | 状态 |
|----|------|
| 仓库 | 公开；含 `valen.train` / `valen.inference` / `valen.evaluate` |
| 权重 | HF `Valen-Preview-0923`（`checkpoint.pt` + `config.json`，需与 Qwen3.5-2B 联载） |
| 数据 | General 100k / Eval 5k / Eval-Game 均在 HF Datasets |
| Demo | HF Space 可交互 |
| 结论 | **已开源**（代码 + 权重 + 数据 + 训练/评测脚本） |

## 核心内容摘要

1. **多模态 System One：** 文本 / 图像 / 视频 state + 任务指令 + 候选 criteria → **不生成答案 token**，经共享 **决策头** 输出 **Choice / Noul / Score** 概率与 **confidence**。
2. **骨干：** Qwen3.5-0.8B 或 **2B**；Preview 为 **2B + vision_top**（LoRA、ViT merger、视觉末 4 层 + 决策头可训，骨干冻结另载）。
3. **训练：** **SFT**（标签监督）→ 可选 **RLCD**（GRPO 式，正确性 + 置信度误差 + KL + 可选 Brier）；stage：`warmup` / `text` / `joint` / `vision_top`。
4. **Preview 血缘：** General 100k SFT 决策头 → Sokoban **30k RLCD × 3 epoch**（`vision_top`）；与 HF `Valen-Sokoban-RLCD-2B` 同 checkpoint。
5. **公开数字（Preview，2026-09）：** 500 步 Sokoban 单步 **87.60%**；100 局完整游戏 **38/100**（35 easy + 65 medium，**模型结果条件筛选** 非无偏全集）；逐步 **~122–128 ms**；9 步通关累计 **1.13 s**（对照 Qwen3.8-27B-FP8 thinking **198.05 s** 且 no-thinking 未通关）。
6. **命名：** Leigh Van Valen **红皇后假说**——持续进化以跟上环境。

## 入口速查

| 路径 / 命令 | 作用 |
|-------------|------|
| `bash scripts/setup/bootstrap.sh` | venv + 依赖 |
| `hf download Valen-Team/Valen-Preview-0923` | Preview checkpoint |
| `python scripts/setup/prepare_model.py` | 拉取并校验 Qwen3.5-2B revision |
| `python -m valen.inference --checkpoint ... --data ...` | JSONL 批量推理 |
| `python -m valen.evaluate --checkpoint ...` | 评测管线 |
| `VALEN_GPUS=8 bash scripts/train/launch.sh configs/train/qwen/*.json` | 多卡 SFT / RLCD |
| `docs/data-format.md` | JSONL 记录格式 |
| `docs/technical.md` | 架构、实验表、Model Card |

## 对 wiki 的映射

- **wiki/entities/valen.md** — 框架/模型实体（与 [typesafe-jev](../../wiki/entities/typesafe-jev.md)、[laya](../../wiki/entities/laya.md) 同谱）
- **wiki/entities/typesafe-jev.md** — 闭源 API 对照；Valen 为 **开源多模态** 延伸
- **wiki/concepts/behavior-tree-vla-orchestration.md** — 毫秒级 typed 分支 + 视觉 state
