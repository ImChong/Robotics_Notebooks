# MiGA: Multi-Gripper-Aware Dataset

> 来源归档

- **标题：** MiGA: Multi-Gripper-Aware Dataset
- **类型：** dataset
- **链接：** <https://huggingface.co/datasets/GVLA/MiGA-Dataset>（索引页；各子集见 [`GVLA` org](https://huggingface.co/GVLA)）
- **论文：** [arXiv:2608.24603](https://arxiv.org/abs/2608.24603) / GVLA
- **项目页：** <https://airvlab.github.io/G-VLA/>
- **许可：** Apache-2.0
- **入库日期：** 2026-09-11
- **一句话说明：** 103K 跨 **5** 类夹爪、**36** 任务、sim+real 的 gripper-aware VLA 数据集；LeRobot Parquet；每帧含 `gripper_id`；强调同任务不同夹爪的策略差异。

---

## 规模与结构

| 维度 | 内容 |
|------|------|
| 轨迹数 | **103,000**（论文）；HF 按 gripper/平台分子集发布 |
| 夹爪类型 | parallel-jaw、3-finger、soft、vacuum/suction、dexterous hand（5 类） |
| 任务 | **36**；四类 Singulated / Stacked / Constrained / Semantic |
| 机器人 | Franka Panda、UR5、UR10、xArm7 等（sim + real） |
| 模态 | 第三人称 RGB `image`、腕部 `wrist_image`、`state` 8D、`actions` 7D delta+夹爪 |
| 标注 | gripper–strategy 对、子步骤、~5% 失败 demo、自然语言 |

### 已发布 HF 子集（示例）

| 子集 | 说明 |
|------|------|
| `GVLA/Franka_panda_parallel_hand_real` | Franka 平行夹爪真机 |
| `GVLA/Franka_inspire_hand_real` | Franka + Inspire 灵巧手真机 |
| `GVLA/Franka_cobot_vacuum_real` | Franka + Cobot 吸盘真机 |
| `GVLA/UR5_robotiq_85_parallel_real` | UR5 + Robotiq 2F-85 真机 |
| `GVLA/UR5_robotiq_3f_3finger_real` | UR5 + Robotiq 三指 |
| `GVLA/UR10_short_cup_vacuum_sim` | UR10 短杯吸盘仿真 |
| `GVLA/Franka_panda_parallel_hand_sim` 等 | 仿真平行夹爪 / 吸盘子集 |

完整表格见 [MiGA-Dataset README](https://huggingface.co/datasets/GVLA/MiGA-Dataset)。

---

## 字段 schema（LeRobot / Parquet）

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | 224×224 RGB | 第三人称 |
| `wrist_image` | 224×224 RGB | 腕部相机 |
| `state` | float32[8] | EE pose + gripper opening |
| `actions` | float32[7] | delta pose + gripper command |
| `gripper_id` | int32 | 夹爪配置 ID（跨子集 stratify / condition） |
| `episode_index` / `frame_index` / `task_index` | int64 | 标准 LeRobot 索引 |

---

## 快速用法

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("GVLA/Franka_panda_parallel_hand_real")
print(ds[0]["gripper_id"], ds[0]["state"].shape)
```

跨夹爪训练：按共享 schema 拼接子集，用 `gripper_id` 或 GVLA 式 soft prompt 做条件化。

---

## 与现有 VLA 数据集对照（论文 Table 1 摘要）

| 数据集 | #Traj | #Gripper | Gripper-specific solution |
|--------|-------|----------|---------------------------|
| OXE | 1M+ | 1 | ✗ |
| DROID | 76K | 1 | ✗ |
| RoboMind | 107K | 2 | ✗ |
| **MiGA** | **103K** | **5** | **✓** |

---

## 对 wiki 的映射

- 论文实体：[GVLA](../../wiki/entities/paper-gvla-gripper-aware-vla.md)
- 论文摘录：[gvla_arxiv_2608_24603.md](../papers/gvla_arxiv_2608_24603.md)
- [LeRobot](../../wiki/entities/lerobot.md) — 数据布局与加载器
