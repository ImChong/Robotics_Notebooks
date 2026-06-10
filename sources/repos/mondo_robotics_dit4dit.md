# Mondo-Robotics/DiT4DiT

> 来源归档（ingest）

- **标题：** DiT4DiT — Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
- **类型：** repo
- **官方入口：** <https://github.com/Mondo-Robotics/DiT4DiT>
- **论文：** <https://arxiv.org/abs/2603.10448>
- **项目页：** <https://dit4dit.github.io/>
- **机构：** Mondo Robotics；HKUST(GZ)；HKUST
- **许可证：** MIT（以仓库 LICENSE 为准）
- **入库日期：** 2026-06-10
- **一句话说明：** DiT4DiT 官方代码：双 DiT Video-Action Model 训练、评测与部署；Cosmos-Predict2.5 视频骨干 + flow-matching 动作头联合优化；含 LIBERO / RoboCasa / Unitree G1 真机管线。

## 仓库要点（公开 README 索引）

| 模块 | 说明 |
|------|------|
| **训练** | Video DiT + Action DiT 联合 dual flow-matching；三时间步 $\tau_v,\tau_f,\tau_a$ |
| **推理** | 可仅动作预测，或同步生成未来视频计划 |
| **部署** | WebSocket policy server；G1 单目 egocentric 闭环 |
| **权重** | 项目页 / Hugging Face 分发（以仓库当前 release 为准） |

## 对 wiki 的映射

- [DiT4DiT 论文实体](../../wiki/entities/paper-dit4dit-video-action-model.md)
- [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) — 同团队后续人形实时 WAM 工作
