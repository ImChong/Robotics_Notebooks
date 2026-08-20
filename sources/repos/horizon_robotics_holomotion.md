# HorizonRobotics / HoloMotion

> 来源归档

- **标题：** HoloMotion（HoloMotion-1）
- **类型：** repo + 模型权重 + 容器镜像 + 技术报告
- **维护方：** Horizon Robotics（地平线）
- **开源模型代码库：** <https://github.com/HorizonRobotics/HoloMotion>
- **项目主页（文档站）：** <https://horizonrobotics.github.io/robot_lab/holomotion>
- **技术报告：** <https://arxiv.org/abs/2605.15336>
- **Hugging Face 模型：** <https://huggingface.co/HorizonRobotics/HoloMotion_models>
- **Docker Hub：** <https://hub.docker.com/r/horizonrobotics/holomotion>
- **入库日期：** 2026-05-18
- **再核日期：** 2026-08-20
- **GitHub：** 约 **634** stars（2026-08-20）
- **一句话说明：** 地平线发布的 **人形全身运动跟踪** 开源栈：配套 **HoloMotion-1** 技术报告中的 **混合大规模运动语料 + 稀疏 MoE Transformer 策略 + 序列级 PPO** 路线，并提供 **HF 权重与官方镜像** 降低复现门槛。
- **沉淀到 wiki：** [`wiki/entities/holomotion.md`](../../wiki/entities/holomotion.md)
- **文档站归档：** [`sources/sites/holomotion-docs.md`](../sites/holomotion-docs.md)

---

## 仓库结构（README / 文档索引，2026-07-22）

| 路径 | 职责 |
|------|------|
| `holosmpl/` | 多源动捕 → 统一 HoloSMPL 表示 |
| `holoretarget/` | HoloRetarget（训练侧高速 / 机上遥操作） |
| `holomotion/` | 策略训练与推理核心 |
| `docs/train_motion_tracking.md` · `evaluate_motion_tracking.md` | 训练 / 评测入口 |
| `docs/realworld_deployment.md` | 离线 motion / 在线 teleop 真机部署 |
| `deployment/` + Docker Hub | **v1.4.1** 镜像与机上部署（离线 motion / 在线 teleop） |
| HF [HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models) · [collections/holomotion](https://huggingface.co/collections/HorizonRobotics/holomotion) | 预训练 motion / velocity tracking 权重 |

## 版本里程碑（README News，2026-08-20）

| 版本 | 要点 |
|------|------|
| **v1.4**（2026-07-16） | **HoloRetarget**：RTX 4090 训练侧 **3000+ FPS**、机上遥操作 **300+ FPS**；**HoloSMPL** 统一 **10+** 数据集/设备 |
| **v1.3**（2026-05-15） | 模型 **60M→0.4B** 参数；语料 **80→2000+ h**；推理 **~100→~300 FPS** |
| **v1.2**（2026-04-04） | 社区可直接部署的 motion / velocity tracking 预训练权重 |

## 路线图（4-Any）

| 阶段 | 目标 | 状态 |
|------|------|------|
| v1 | **Any Pose** — 多样全身模仿跟踪 | ✅ 已完成 |
| v2 | **Any Command** — 语言/任务条件运动生成 | 🚀 下一步 |
| v3 | **Any Embodiment** — 跨形态泛化 | 🧭 规划中 |
| v4 | **Any Terrain** — 复杂地形适应 | 🧭 规划中 |

## 下游复用（README Projects Using HoloMotion）

| 项目 | 关系 |
|------|------|
| [OMG](https://github.com/Tsinghua-MARS-Lab/OMG) | omni-modal 生成器 + HoloMotion tracker |
| [HoloAgent-0](https://github.com/HorizonRobotics/HoloAgent) | Embodied AgentOS 中的全身运动技能层 |

## 与「robot_lab」命名的关系

文档站点路径含 `robot_lab/holomotion`，指 **Horizon Robotics 组织下托管的 GitHub Pages**，与社区维护的 IsaacLab 扩展仓库 **[fan-ziqi/robot_lab](robot_lab.md)** 不是同一项目；阅读文档与引用链接时建议 **以组织名与域名区分**。

---

## 资料在知识库中的角色

| 资料 | 角色 |
|------|------|
| [holomotion_arxiv_2605_15336.md](../papers/holomotion_arxiv_2605_15336.md) | 方法、数据与系统叙述的论文级摘录 |
| 本文件 | 官方入口（代码 / 站点 / 权重 / 容器）一站式索引 |
