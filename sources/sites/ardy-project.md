# ARDY — NVIDIA 官方项目页

- **来源：** <https://research.nvidia.com/labs/sil/projects/ardy/>
- **类型：** site（项目页 / 交互演示）
- **机构：** NVIDIA Research（SIL）· ETH Zürich
- **归档日期：** 2026-07-11（更新 2026-09-06）
- **论文：** ACM TOG · SIGGRAPH 2026 · DOI [10.1145/3811284](https://doi.org/10.1145/3811284)
- **arXiv：** <https://arxiv.org/abs/2607.08741>
- **PDF：** <https://research.nvidia.com/labs/sil/projects/ardy/assets/ardy_paper.pdf>
- **代码：** <https://github.com/nv-tlabs/ardy>
- **模型：** <https://huggingface.co/collections/nvidia/ardy>

## 一句话说明

**ARDY** 是面向 **交互应用** 的 **自回归扩散** 人体运动生成框架：支持 **流式文本提示** 与 **灵活长时域运动学约束**（根部轨迹/路点、全身关键帧、末端关节位姿/旋转及组合），在 **实时响应** 下生成高保真 3D 人体运动。

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy)（Apache-2.0） |
| **权重** | **已发布** — HF Collection [nvidia/ardy](https://huggingface.co/collections/nvidia/ardy)（Core/G1 多 horizon；NVIDIA Open Model） |
| **交互 Demo** | 项目页 + 仓库 `run_demo.py`（Viser 浏览器 UI） |

## 核心能力（项目页归纳）

| 能力 | 示例 |
|------|------|
| 在线文本 | Limp、Pick & Put、Stealthy Walk、Victory Dance、**Prompt Streaming** |
| 根部约束 | 稠密轨迹、稀疏路点、**超出当前窗口** 的长时域目标 |
| 全身/稀疏关节 | 全身关键帧、末端位置/朝向、约束链与组合 |
| 交互 locomotion | 鼠标路点 + 键盘速度指令的实时行走控制 |
| 人形下游 | ARDY + **SONIC** → **Unitree G1**（机器人芭蕾等） |

## 架构要点（Method）

- **混合表示：** 显式 **global root** + **潜空间 body embedding**（Motion Tokenizer）。
- **自回归两阶段去噪：** 可变历史；**root 先于 body**；mask 化长时域约束。
- **无限流式：** 项目页强调 **分钟级** 连续生成。

## Hugging Face 模型（2026-09-06）

Collection 含 Core / G1 等 checkpoint（详见 [nv_tlabs_ardy.md](../repos/nv_tlabs_ardy.md) 表）：Horizon8 利于 **快速 prompt 切换**；Horizon40/52 利于 **质量与长上下文**。

## NVIDIA 人形运动生态（项目页互链）

| 组件 | 与 ARDY 关系 |
|------|----------------|
| [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) | 离线可控扩散姊妹 |
| [MotionBricks](https://research.nvidia.com/labs/sil/projects/motionbricks/) | 模块化实时 API |
| [GEAR SONIC](https://nvlabs.github.io/GEAR-SONIC/) | 生成→G1 物理跟踪 |
| [ProtoMotions](https://protomotions.github.io/) | 物理策略训练 |

## 对 wiki 的映射

1. **[ARDY（实体页）](../../wiki/entities/ardy.md)**
2. **[Kimodo](../../wiki/entities/kimodo.md)** — 离线对照
3. **[nv-tlabs/ardy 仓库](../../sources/repos/nv_tlabs_ardy.md)**
