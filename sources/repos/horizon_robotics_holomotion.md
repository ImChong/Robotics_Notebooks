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
- **再核日期：** 2026-07-22
- **GitHub：** 约 **590** stars（2026-07-22）
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
| `deployment/` + Docker Hub | v1.4 镜像与机上部署 |
| HF `HorizonRobotics/HoloMotion_models` | 预训练 motion / velocity tracking 权重 |

## 与「robot_lab」命名的关系

文档站点路径含 `robot_lab/holomotion`，指 **Horizon Robotics 组织下托管的 GitHub Pages**，与社区维护的 IsaacLab 扩展仓库 **[fan-ziqi/robot_lab](robot_lab.md)** 不是同一项目；阅读文档与引用链接时建议 **以组织名与域名区分**。

---

## 资料在知识库中的角色

| 资料 | 角色 |
|------|------|
| [holomotion_arxiv_2605_15336.md](../papers/holomotion_arxiv_2605_15336.md) | 方法、数据与系统叙述的论文级摘录 |
| 本文件 | 官方入口（代码 / 站点 / 权重 / 容器）一站式索引 |
