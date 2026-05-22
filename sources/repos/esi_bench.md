# ESI-Bench（官方实现与数据）

> 来源归档

- **标题：** ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop
- **类型：** repo + benchmark toolkit + HF dataset
- **组织：** Stanford / UCLA / Northwestern 等（见论文）
- **代码：** <https://github.com/ESI-Bench/ESI-Bench>
- **论文：** <https://arxiv.org/abs/2605.18746>
- **项目页：** <https://esi-bench.github.io/>
- **Hugging Face 数据：** <https://huggingface.co/datasets/esi-bench/ESI-Bench>
- **许可证：** MIT（README badge）
- **入库日期：** 2026-05-22
- **一句话说明：** 开源 **主动探索** 运行器 + **dataset/json_clean** 任务 JSON + **dataset_generation** 构造脚本；依赖 **`behavior` conda 环境** 与本地 **OmniGibson / BEHAVIOR-1K** 资产，逐步截图后调用 **OpenAI / Gemini** API 写 `answer.json`。

## 仓库结构（README 归纳）

```
esi-bench/
├── dataset/json_clean/          # 按任务类分目录的 question JSON
├── src/active_explore/          # OmniGibson 场景 + 逐步图像 + MLLM + answer.json
├── src/dataset_generation/      # 数据集构造
└── outputs/                     # 结果与 step 图像（gitignore）
```

**任务 JSON 顶层目录名（与论文类对应）：** Action Sequencing、Cognitive Mapping、Enumerative Perception、Metric Comparison、Perceptual Grounding、Physical Dynamics、Physical Structure（站页 Physical Capacity）、Spatial Relations、Specular Reflection、Temporal Understanding（站页 Temporal Scene）。

## 环境与运行（摘要）

1. `conda activate behavior`（README 假定已有 BEHAVIOR / OmniGibson 安装）。
2. 设置 `OPENAI_API_KEY` 或 `GEMINI_API_KEY`。
3. 示例：`python src/main.py --task counting --metadata dataset/json_clean/.../q_000.json --provider gemini --model gemini-3.1-pro-preview --max-steps 30 ...`

## 对 wiki 的映射

- 实体页：[ESI-Bench](../../wiki/entities/esi-bench.md) — 复现门槛、目录与评测范式。
- 交叉： [3D 空间 VQA](../../wiki/concepts/3d-spatial-vqa.md)、[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)、[WEM / HTEWorld](../../wiki/entities/paper-wem-world-ego-modeling.md)（同属 BEHAVIOR-1K / OmniGibson 生态）。
