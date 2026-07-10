# Evo-1（MINT-SJTU/Evo-1）

> 来源归档

- **标题：** Evo-1 — Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
- **类型：** repo
- **组织：** MINT-SJTU（上海交通大学）
- **代码：** <https://github.com/MINT-SJTU/Evo-1>
- **项目页：** <https://mint-sjtu.github.io/Evo-1.io/>
- **论文：** <https://arxiv.org/abs/2511.04555>（**CVPR 2026**，Efficient CVPR Badge）
- **入库日期：** 2026-07-10
- **一句话说明：** Evo-1 官方实现：InternVL3-1B + cross-modulated DiT flow-matching、两阶段训练脚本、Meta-World/LIBERO/RoboTwin 评测、xArm6 推理服务、**LeRobot v2.1 数据训练** 与 **SO100/SO101 `lerobot-record` 部署**（已并入官方 LeRobot）。
- **沉淀到 wiki：** [Evo-1（论文实体）](../../wiki/entities/paper-evo1-lightweight-vla.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **0.77B 轻量 flow-VLA**；**无机器人预训练** 的 SOTA 级仿真与真机结果 |
| [LeRobot](../../wiki/entities/lerobot.md) | 训练数据 **LeRobot v2.1**；2026-07 起 **官方 LeRobot 内置 Evo-1 策略** |
| [MINT](../../wiki/entities/paper-mint-vla.md) | 同 **MINT-SJTU** 研究线；MINT 走频域意图分词，Evo-1 走 **语义保持轻量 flow** |
| [Manipulation](../../wiki/tasks/manipulation.md) | Meta-World / LIBERO / RoboTwin / xArm6 操作评测语境 |
| [Action Chunking](../../wiki/methods/action-chunking.md) | 动作专家输出 **H 步连续 action chunk** |

## 工程要点（README 摘要）

- **分支：** `main`（核心训练/评测）、`evo1-lerobot`（LeRobot 全集成）、`evo1-flash`（更快训练、更低显存）。
- **权重：** `MINT-SJTU/Evo1_MetaWorld`、`Evo1_LIBERO`、`Evo1_SO100`（Hugging Face）。
- **真机：** WebSocket `Evo1_server.py` + 各平台 client；SO100/SO101 走 `lerobot-record --policy.path=...`。
- **依赖：** FlashAttention 对成功率/运动稳定性影响显著（README 强调必装）。

## 为何值得保留

- **轻量 VLA 可复现入口：** 完整两阶段训练、仿真评测与 **LeRobot 一条命令真机闭环**。
- 与 [Humanoid_Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 单篇深读互补：本库负责 **跨主题轻量 VLA 选型与部署索引**。
