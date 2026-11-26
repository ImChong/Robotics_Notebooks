---
type: entity
tags: [nvidia, research-lab, spatial-intelligence, 3d-perception, world-models, neural-reconstruction]
status: complete
updated: 2026-11-26
related:
  - ./paper-axolotl3d.md
  - ./nvidia-gear-lab.md
  - ./paper-instant-nurec.md
  - ./nvidia-nurec.md
  - ./kimodo.md
  - ./paper-gamma-world-multi-agent.md
  - ../methods/ase.md
  - ../methods/generative-world-models.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../concepts/software-in-the-loop.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/sites/nvidia-spatial-intelligence-lab.md
  - ../../sources/repos/nv_tlabs.md
original:
  - ../../sources/sites/nvidia-spatial-intelligence-lab.md
summary: "NVIDIA Spatial Intelligence Lab（SIL）是 NVIDIA Research 的空间智能研究组：推进 AI 感知、建模并与物理世界交互；公开代码多挂在 GitHub nv-tlabs，代表项目含 Instant NuRec、Kimodo、Gamma-World、ASE 等。"
---

# NVIDIA 空间智能实验室（NVIDIA Spatial Intelligence Lab, SIL）

**NVIDIA Spatial Intelligence Lab**（门户：<https://research.nvidia.com/labs/sil/>，GitHub：<https://github.com/nv-tlabs>）是 NVIDIA Research 下的 **空间智能** 研究组：目标是把 AI 的 **感知、世界建模与物理交互** 推到可部署的基础技术，而不是只做 2D 识别或纯语言推理。

## 一句话定义

**NVIDIA 的空间智能基础研究组：3D/4D 感知、神经重建与生成式世界模型多从 `nv-tlabs` 开源，项目页挂在 `research.nvidia.com/labs/sil/projects/*`——与工程术语 Software-in-the-Loop（SIL）同名不同义。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SIL | Spatial Intelligence Lab | 本页所述 NVIDIA 研究组（勿与 Software-in-the-Loop 混淆） |
| 3DGS | 3D Gaussian Splatting | 神经场景表示；Instant NuRec 等输出形态 |
| WFM | World Foundation Model | 视频/世界生成基础模型；Gamma-World 等多智能体路线 |
| NuRec | NVIDIA Neural Reconstruction | 驾驶/机器人神经重建产品栈 |
| ViT | Vision Transformer | 多视图编码骨干（Instant NuRec 等） |
| ASE | Adversarial Skill Embeddings | nv-tlabs 技能嵌入 RL 框架 |
| Real2Sim | Real to Simulation | 真实日志 → 可仿真资产；SIL 实验室与 NuRec 主线 |

## 为什么重要

- **Real2Sim / 仿真评测上游：** [Instant NuRec](./paper-instant-nurec.md) 把多相机驾驶片段 **秒级** 重建成可仿真 3DGS；在作者评估的五个策略上，AlpaSim 排序与优化版 NuRec 一致，但碰撞率和驶离道路率并不相同——直接服务 [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) 与 [NuRec](./nvidia-nurec.md) 产品栈。
- **机器人数据上游：** [Kimodo](./kimodo.md) 从大规模动捕生成 **G1 / SOMA** 参考轨迹，衔接 [ProtoMotions](./protomotions.md) 与 [SONIC](./nvidia-sonic.md) 跟踪。
- **生成式世界模型：** [Gamma-World](./paper-gamma-world-multi-agent.md) 等多体可控视频世界模型补 [生成式世界模型](../methods/generative-world-models.md) 的 **空间交互** 维度。
- **遮挡感知 3D 补全：** [Axolotl3D](./paper-axolotl3d.md) 在 Hunyuan3D-DiT 上统一 **部分点云 + 多视图 + 编辑** 条件，服务 Real2Sim 与场景物体补全（核对时官方项目页未公开独立源码入口）。
- **与 GEAR 互补：** [GEAR Lab](./nvidia-gear-lab.md) 偏通才具身 agent / GR00T / SONIC **系统栈**；SIL 更偏 **几何、重建、4D 与生成式空间先验**。

## 核心机制：按输入和产物区分研究方向

官方将研究分为世界仿真、4D 感知、3D/4D 内容创建与编辑，并单列数据 SDK、工具和源码发布。下图按项目的输入与输出组织，不表示这些项目共享模型或必须串联。

```mermaid
flowchart TB
  DRIVE["标定后的多相机驾驶日志"] --> REC["Instant NuRec：前馈神经重建"]
  REC --> WORLD["静态与动态 3DGS、天空等分层场景"]
  WORLD --> SIM["AlpaSim：闭环策略评测"]
  ACTION["每个智能体的动作"] --> GEN["Gamma-World：交互式世界生成"]
  GEN --> ROLLOUT["同一场景的多视角视频推演"]
  PROMPT["文本、关键帧或路径约束"] --> MOT["Kimodo：运动学扩散"]
  MOT --> TRAJ["SOMA / G1 骨架参考轨迹"]
  TRAJ --> TRACK["ProtoMotions 或 GEAR-SONIC：物理跟踪训练或仿真"]
```

### 代码与项目页怎么找

| 入口 | 用途 |
|------|------|
| [research.nvidia.com/labs/sil/](https://research.nvidia.com/labs/sil/) | 实验室门户与 `projects/*` 交互 demo |
| [github.com/nv-tlabs](https://github.com/nv-tlabs) | 实验室公开实现的入口之一，部分项目也在 NVIDIA 等组织发布 |
| 各论文 `sources/repos/` | 本库逐步归档的 README 与开源边界 |

## 本库已索引代表工作

| 工作 | 本库页 |
|------|--------|
| Instant NuRec | [paper-instant-nurec](./paper-instant-nurec.md) |
| Omniverse NuRec | [nvidia-nurec](./nvidia-nurec.md) |
| Kimodo | [kimodo](./kimodo.md) |
| Gamma-World | [paper-gamma-world-multi-agent](./paper-gamma-world-multi-agent.md) |
| Axolotl3D | [paper-axolotl3d](./paper-axolotl3d.md) |
| ASE | [ASE](../methods/ase.md) |
| ChronoEdit | [paper-sa-2510-04290-chronoedit](../entities/paper-sa-2510-04290-chronoedit-towards-temporal-reasoning-for-image.md) |

完整高星仓清单见 [nv-tlabs 归档](../../sources/repos/nv_tlabs.md)。

## 工程实践

| 目标 | 建议 |
|------|------|
| 查某 SIL 论文代码 | 项目页的 Code / GitHub 链接；同时搜 `nv-tlabs` 与 `NVIDIA/` |
| 驾驶 Real2Sim | 从标定过的驾驶日志重建；需要时做逐场景优化，再在 AlpaSim 单独验证策略绝对风险指标 |
| 人形运动数据 | 用 Kimodo 生成目标骨架轨迹，再用 ProtoMotions 或 GEAR-SONIC 等验证动态可行性 |
| 别混淆 SIL | 仿真 **Software-in-the-Loop** 见 [概念页](../concepts/software-in-the-loop.md) |

**开源与运行时边界（2026-11-26）：** 门户是研究导航，不是可安装的统一软件，因此本页没有实验室级源码运行时序。模型、数据和许可证逐项目核对，实际运行入口见对应实体页。

## 局限与风险

- **缩写冲突：** 站内 **SIL** 亦指 [Software-in-the-Loop](../concepts/software-in-the-loop.md)——读文献/课程时先确认语境。
- **组织分裂：** 部分项目（如 Instant NuRec）在 **`NVIDIA/`** 而非 `nv-tlabs`。
- **产品 vs 研究：** NuRec 容器、Cosmos 等与实验室论文 **发布节奏不同**，勿把 demo 页当生产 SLA。

## 关联页面

- [NVIDIA GEAR Lab](./nvidia-gear-lab.md) — 具身通才 agent 研究组
- [NVIDIA NuRec](./nvidia-nurec.md) — 产品化神经重建
- [Software-in-the-Loop（概念）](../concepts/software-in-the-loop.md) — Isaac Sim 工程 SIL
- [生成式世界模型](../methods/generative-world-models.md)
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — SIL 的 3D/4D 感知与重建落到感知栈哪一层

## 参考来源

- [SIL 实验室门户归档](../../sources/sites/nvidia-spatial-intelligence-lab.md)
- [nv-tlabs GitHub 归档](../../sources/repos/nv_tlabs.md)

## 推荐继续阅读

- [NVIDIA Spatial Intelligence Lab](https://research.nvidia.com/labs/sil/)
- [nv-tlabs（GitHub）](https://github.com/nv-tlabs)
- [Instant NuRec 项目页](https://research.nvidia.com/labs/sil/projects/instant-nurec/)
- [Kimodo 项目页](https://research.nvidia.com/labs/sil/projects/kimodo/)
