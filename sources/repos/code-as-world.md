# Code-as-World 官方仓库

> 来源归档（repo）

- **标题：** Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning
- **类型：** repo
- **链接：** https://github.com/MirroS-Lab/Code-as-World
- **arXiv：** <https://arxiv.org/abs/2608.27549>
- **项目页：** <https://mirros-lab.github.io/code-as-world/>
- **入库日期：** 2026-09-21
- **一句话说明：** 官方开源：Code-as-World-VL 4B/9B 本地推理、QuantiPhy 评测与 MuJoCo simulation 示例。
- **沉淀到 wiki：** [`wiki/entities/paper-code-as-world.md`](../../wiki/entities/paper-code-as-world.md)

## 开源状态（步骤 2.5，2026-09-21）

- **已开源（推理侧）：** `requirements/inference.txt`；`python -m code_as_world.evaluation {4b|9b}`；vLLM OpenAI-compatible serving；`python -m code_as_world.simulation`（MuJoCo 弹道足球例）。
- **权重：** `hf download MirroS-Lab/Code-as-World-VL-4B|9B`。
- **外部依赖：** QuantiPhy 仓库 + `PaulineLi/QuantiPhy-validation` 数据集用于官方 eval 复现。

## README 入口摘要

| 路径 | 用途 |
|------|------|
| `code_as_world/evaluation` | QuantiPhy 批量推理与 metric summary |
| `code_as_world/simulation` | Video-driven abstraction / MuJoCo 示例 |
| `code_as_world/templates/` | vLLM chat template（Qwen3.5 no-think） |
| `requirements/inference.txt` | CUDA 推理依赖 |
| `requirements/simulation.txt` | MuJoCo simulation 依赖 |
