# HadiZayer/masked-visual-actions

> 来源归档

- **标题：** Masked Visual Actions（官方实现）
- **类型：** repo
- **组织 / 作者：** HadiZayer
- **代码：** <https://github.com/HadiZayer/masked-visual-actions>
- **权重：** <https://huggingface.co/HadiZayer/masked-visual-actions>
- **基座：** ModelScope [`PAI/Wan2.2-Fun-A14B-Control`](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control)（经 DiffSynth-Studio）
- **论文：** <https://arxiv.org/abs/2607.19343>
- **项目页：** <https://masked-visual-actions.github.io>
- **入库日期：** 2026-07-22
- **一句话说明：** 在 **Wan2.2-Fun-A14B-Control** 上训 **双专家 LoRA**（高噪声 / 低噪声 DiT）的薄封装：不改视频模型本身，用 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)（钉定 commit `3743b130…`）做训练与推理；输入 **control video**（如 URDF 渲染机器人）+ **reference image** + **text prompt**，输出 RGB 视频。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `git checkout 3743b1307caf2562af60d475b22d4b6be68e7cd0`（DiffSynth-Studio） | 钉定训练/推理依赖 |
| `python inference/download_weights.py --out ./checkpoints` | 拉取 HF 双 LoRA |
| `python inference/infer.py --lora-high … --lora-low … --control-video … [--reference-image …] --prompt … --output out.mp4` | 推理生成 |
| `infer.py --low-vram` | 磁盘 offload，降峰值显存 |
| `DATASET_CSV=… OUTPUT_DIR=… bash training/train_control.sh` | 在 DiffSynth 根目录训高/低噪声 LoRA（rank 256，默认 4 GPU） |
| URDF 渲染工具 | README：**coming soon** |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) | 实体归纳：掩码动作接口、前向/逆统一、策略评估 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 像素域动作条件 WM；条件是 **掩码轨迹** 而非低维关节 |
| [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) | Best-of-N 规划 + 离线/真机策略评估沙盒 |
| [world-models-route-01-cascade](../../wiki/overview/world-models-route-01-cascade.md) | 逆设定 + IDM 抽动作，对齐级联「预测→动作解码」 |
| [DriftWorld](../../wiki/entities/paper-driftworld.md) / [OSCAR](../../wiki/entities/paper-oscar.md) | 同属动作条件视频 WM + 虚拟评估；条件表示与速度卖点不同 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/masked_visual_actions_arxiv_2607_19343.md`](../papers/masked_visual_actions_arxiv_2607_19343.md)
- 项目页：[`sources/sites/masked-visual-actions-github-io.md`](../sites/masked-visual-actions-github-io.md)
- 沉淀 **[`wiki/entities/paper-masked-visual-actions.md`](../../wiki/entities/paper-masked-visual-actions.md)**
