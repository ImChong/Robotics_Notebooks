---
type: entity
tags: [repo, unitree, unitreerobotics, world-model, foundation-model, imitation-learning]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unifolm-vla.md
  - ./unitree-lerobot.md
  - ./z1-sdk.md
  - ../concepts/world-action-models.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/repos/unifolm-world-model-action.md
  - ../../sources/repos/unitree.md
summary: "UnifoLM-WMA-0 是宇树开源的世界模型–动作架构：世界模型既作交互式仿真引擎生成数据，也与动作头结合做策略增强；训练/推理/权重/部署均已开源。"
---

# UnifoLM-WMA-0（unifolm-world-model-action）

**UnifoLM-WMA-0** 是面向通用机器人学习的 **World-Model–Action** 架构：核心世界模型理解机器人与环境的物理交互，并提供 **仿真引擎** 与 **策略增强** 两种用法。

## 一句话定义

用世界模型预测未来交互，既合成数据又辅助动作头决策；官方开源到部署代码，覆盖 Z1 / G1 等多项开源数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WMA | World-Model-Action | 世界模型–动作架构 |
| VLA | Vision-Language-Action | 同家族的另一路线 |
| HF | Hugging Face | 权重与数据托管 |
| IL | Imitation Learning | 常见训练设定 |
| Open-X | Open X-Embodiment | Base 模型微调数据来源之一 |
| G1 | Unitree G1 Humanoid | 真机演示平台之一 |

## 为什么重要

- 把「世界模型当模拟器」与「世界模型助决策」写进同一官方框架，便于对照学术 WAM/世界模型工作。
- Open-Source Plan：**Training / Inference / Checkpoints / Deployment** 齐全。
- 真机演示覆盖 Z1 堆叠、清理与 G1 装相机等，连接机械臂与人形两条硬件线。

## 核心原理

| 功能 | 说明 |
|------|------|
| Simulation Engine | 交互式仿真，生成学习用合成数据 |
| Policy Enhancement | 接动作头，借未来交互预测优化决策 |

**权重**：`UnifoLM-WMA-0_Base`（Open-X）；`UnifoLM-WMA-0_Dual`（Unitree 开源数据，决策+仿真模式）。

**实验数据（节选）**：Z1_StackBox、Z1 双臂 StackBox / Cleanup_Pencils、G1_Pack_Camera 等（HF `unitreerobotics`）。

## 工程实践

```bash
conda create -n unifolm-wma python==3.10.18 && conda activate unifolm-wma
conda install pinocchio=3.2.0 ffmpeg=7.1.1 -c conda-forge -y
git clone --recurse-submodules https://github.com/unitreerobotics/unifolm-world-model-action.git
cd unifolm-world-model-action && pip install -e .
cd external/dlimp && pip install -e .
```

项目页：<https://unigen-x.github.io/unifolm-world-model-action.github.io>。

## 局限与风险

- 子模块与 pinocchio/ffmpeg 版本钉扎，环境复现成本高。
- 演示窗口中的「未来动作视频」是世界模型预测，不等于真机必然成功。
- 与 VLA 仓分工不同：选 WMA 还是 VLA 取决于是否需要世界模型仿真/增强，而不是简单比星标。

## 关联页面

- [UnifoLM-VLA](./unifolm-vla.md)
- [World-Action Models](../concepts/world-action-models.md)
- [Z1 软件栈](./z1-sdk.md)
- [unitree_lerobot](./unitree-lerobot.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unifolm-world-model-action.md](../../sources/repos/unifolm-world-model-action.md)
- 上游：<https://github.com/unitreerobotics/unifolm-world-model-action>

## 推荐继续阅读

- HF Collections：<https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c>

