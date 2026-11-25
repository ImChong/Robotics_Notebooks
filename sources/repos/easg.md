# EASG 公开源码归档

- **项目：** Explicit Action-Space Grounding
- **代码：** https://github.com/General-Robo/EASG
- **源码快照：** [1d0196e](https://github.com/General-Robo/EASG/tree/1d0196e39db9103aafedd61842883d925b052dd3)
- **日期：** 2026-11-25
- **原文入口：** [arXiv v1](../papers/easg_arxiv_2610_12832.md) · [项目页](../sites/easg-project.md)
- **核对范围：** 根 README、Isaac Sim 主循环及解析/动作模块；LIBERO 转换、LoRA 训练和分进程评测入口、四套件适配器目录与测试。未做 GPU 仿真、在线付费推理或训练。
- **发布范围：** 公开可读源码和 LIBERO 的 Goal、Spatial、Object、Long 四个 LoRA 适配器；未见真机微调权重或真机示范数据、仓库根目录无独立许可证。

| 场景 | 真实入口 | 链路 |
|---|---|---|
| Isaac Sim 零样本 | `script/run.py` | `capture` / `prompt_for` → 可选 `VisionClient.chat` → `parse_vlm_motion` 与动作限幅 → TCP 到手部目标 → `RobotController` → `scene.evaluate` |
| LIBERO 数据/训练 | `libero-finetune/scripts/convert_libero_hdf5.py`、`validate_dataset.py`、`train.sh` | HDF5 → 监督样本 → `finetune.py` 读取套件 recipe，微调 Qwen3.5-0.8B 文本生成（学习率 5e-5，90 epoch，单卡 batch 4 × 梯度累积 2） |
| LIBERO 评估 | `libero-finetune/scripts/evaluate.py` | `QwenPolicyWorker` 通过模型子进程加载 LoRA，生成前缀受限文本；仿真侧解析、逆缩放、推进环境，输出 `summary.json` |

- Isaac Sim 5.1.0 的公开默认用 OpenRouter `qwen/qwen3.8-27b`；`--observe-only` 检查图像/提示词时不访问模型。在线会上传图像和提示词并计费；README 的约 55% 仅是此仿真示例口径。
- 原论文真机基线与教学仓库仿真并非同一环境。repo 的 LIBERO 默认 16 回合/任务/次运行，不应直接当作原论文的三 seed 平均。
