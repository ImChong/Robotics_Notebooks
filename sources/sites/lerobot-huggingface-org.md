# LeRobot（Hugging Face 组织页）

> 来源归档

- **标题：** LeRobot — Hugging Face Organization
- **类型：** site（Hugging Face 组织主页）
- **机构：** Hugging Face
- **链接：** https://huggingface.co/lerobot
- **代码仓：** https://github.com/huggingface/lerobot
- **入库日期：** 2026-07-12
- **一句话说明：** LeRobot 在 Hugging Face Hub 上的**官方模型 / 数据集 / Spaces 分发入口**；与 GitHub 代码仓互补，承载预训练权重、社区演示数据、可视化工具与教程论文索引。

## 为什么值得保留

- **代码 vs Hub 分工：** GitHub 仓负责 **PyTorch 库、CLI、硬件驱动与训练脚本**；`huggingface.co/lerobot` 负责 **可 `from_pretrained` 拉取的策略权重、LeRobot 格式数据集、交互式 Spaces**——工程上常「装库从 GitHub、拉权重从 Hub」。
- **社区规模信号（2026-07-12 页面快照）：** 约 **56** 个模型、**187** 个数据集、**11** 个 Collections、**9** 个 Spaces、**1** 个公开 bucket（`lerobot/robot-urdfs`，~187 MB）。
- **降低入门摩擦：** 组织卡写明提供 **预训练模型、人类演示数据集与仿真环境**，目标是把机器人 ML 门槛降到「人人可贡献、可复用」。

## 组织定位（Organization Card 摘录）

> State-of-the-art machine learning for real-world robotics.
>
> LeRobot aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier for entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.
>
> LeRobot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning.

关联论文入口：**Robot Learning: A Tutorial**（组织页 Papers 区列出）。

## Hub 资产分层

| 层级 | 规模（2026-07） | 典型用途 |
|------|-----------------|----------|
| **Models** | ~56 | `lerobot-*` 策略 checkpoint；`pipeline_tag: robotics`、`library_name: lerobot` 便于 Hub 筛选 |
| **Datasets** | ~187 | LeRobot v2.0+ 演示数据；社区与官方采集任务 |
| **Collections** | 11 | 按方法族打包权重，如 **FastWAM**、**VLA-JEPA** |
| **Spaces** | 9 | 无本地 GPU 时的可视化、基准与教程入口 |
| **Buckets** | 1 | `lerobot/robot-urdfs` — 机器人 URDF 资产 |

## 代表性 Collections 与模型族（近期更新）

| 系列 | 代表 repo id | 备注 |
|------|--------------|------|
| **π₀ / π₀.₅** | `lerobot/pi0_base`、`lerobot/pi05_base`、`lerobot/pi0fast-base` | VLA 系预训练；Hub 下载量高（`pi05_base` 万级） |
| **VLA-JEPA** | `lerobot/VLA-JEPA-LIBERO`、`VLA-JEPA-Pretrain`、`VLA-JEPA-SimplerEnv` | 3B 级；arXiv:2602.10098 |
| **FastWAM** | `lerobot/fastwam_base` 及 LIBERO / RoboTwin 变体 | 6B 级世界–动作模型；arXiv:2603.16666 |
| **LingBot-VA** | `lerobot/lingbot_va_base`、`lingbot_va_libero_long`、`lingbot_va_robotwin` | 与 [LingBot-VLA 2.0](../repos/lingbot-vla-v2.md) 生态交叉 |
| **MolmoAct2** | `lerobot/MolmoAct2-SO100_101-LeRobot` 等 | SO100/101、LIBERO、DROID 等平台化 checkpoint |
| **任务示范** | `lerobot/folding_latest` | 叠衣等端到端真机策略示例 |

> 完整列表以 Hub 为准；上表仅摘录 ingest 日 API / 页面可见的高频族。

## 代表性 Spaces

| Space | 用途 |
|-------|------|
| **LeLab** | 浏览器内简易 LeRobot 交互入口 |
| **Visualize Dataset (v2.0+)** | LeRobot 数据集交互式可视化（相机轨迹、动作曲线等） |
| **video-benchmark** | 视频编解码性能基准 |
| **LeRobot Brand Assets** | 官方 logo 与品牌素材 |
| **Unfolding Robotics** | 开源叠衣「数据→部署」指南演示 |

## 与 GitHub 仓的关系

```
GitHub huggingface/lerobot          Hugging Face org lerobot
├── Python 包 / CLI                  ├── Models（策略权重）
├── lerobot-record / train           ├── Datasets（演示数据）
├── 硬件驱动与仿真 glue              ├── Spaces（可视化 / 教程）
└── 策略实现源码                     └── Collections / Buckets（打包分发）
         │                                      │
         └──────── policy.path / from_pretrained ────────┘
```

- **训练产出上传 Hub** → 他人 `lerobot-record --policy.path=lerobot/...` 或 `PreTrainedPolicy.from_pretrained` 复现。
- **数据集** 常以 `lerobot/<task>_<robot>` 命名；v2.0+ 格式可用 Visualize Dataset Space 快速质检。

## 对 wiki 的映射

- 实体页：[lerobot](../../wiki/entities/lerobot.md) — 补充 **Hub 分发层** 与模型 / 数据 / Spaces 选型
- 方法交叉：[vla](../../wiki/methods/vla.md)、[imitation-learning](../../wiki/methods/imitation-learning.md)
- 相关实体：[lingbot-vla-v2](../../wiki/entities/lingbot-vla-v2.md)、[openvla](../../wiki/entities/openvla.md)、[paper-evo1-lightweight-vla](../../wiki/entities/paper-evo1-lightweight-vla.md)
- 代码归档：[lerobot.md](../repos/lerobot.md)

## 参考来源（原始）

- Hugging Face 组织页：<https://huggingface.co/lerobot>
- GitHub 代码仓：<https://github.com/huggingface/lerobot>
