# MGDP（arclab-hku/MGDP）— 原始资料归档

- **来源：** <https://github.com/arclab-hku/MGDP>
- **类型：** repo
- **论文：** [MGDP（Advanced Science 2026）](https://doi.org/10.1002/advs.202524345)
- **项目页：** <https://arclab-hku.github.io/MGDP/>
- **机构：** 香港大学（HKU）ARC Lab；北京理工大学（BIT）
- **归档日期：** 2026-09-15
- **默认分支：** `master`

## 一句话说明

**MGDP** 官方实现：Isaac Gym 四足感知 DRL；**NVIDIA Warp**（`warp_sensor`）并行深度；**Stage1** 对比学习深度感知模型 + **Stage2** 行走控制器微调；支持多 `DOG_NAMES` 混训与深度/高程可视化。

## 开源核查（步骤 2.5）

| 项 | 结论（截至 2026-09-15） |
|----|-------------------------|
| **代码** | **已开源** — `legged_gym/scripts/train.py`、`resume.py`、`vis_stage1.py`、`vis_stage2.py` |
| **仿真** | 内嵌 **Isaac Gym**（需 NVIDIA 账号单独许可）；Python **3.8.20** + PyTorch **1.10** cu113 |
| **深度传感器** | `warp_sensor` 子包（`pip install -e .`；`warp-cam` 测试） |
| **预训练权重** | README **未提供** HF/网盘链接；需按 Stage1→Stage2 自行训练 |
| **许可证** | 仓库根 **无** 独立 `LICENSE` 文件；使用须遵守 Isaac Gym 许可 |

## 训练入口（README）

| 阶段 | 命令 | 说明 |
|------|------|------|
| Stage 1 | `python legged_gym/scripts/train.py` | 通用深度感知模型；`args.task` 如 `random_dog_stage1` |
| Stage 2 | `python legged_gym/scripts/resume.py` | 行走控制器；`args.resume_name` + `random_dog_stage2`；可设 `DOG_NAMES` 多狗混训 |
| 可视化 | `vis_stage1.py` / `vis_stage2.py` | 噪声/预测/干净深度与高程对比 |
| 地形浏览 | `play_terrain.py` | 极端地形预览 |

## 目录要点

| 路径 | 作用 |
|------|------|
| `legged_gym/legged_gym/envs/` | 任务与环境定义 |
| `legged_gym/rl/MGDP/` | MGDP 算法与模型 |
| `legged_gym/models/MGDP/` | 模型权重保存路径 |
| `warp_sensor/` | Warp 并行相机/深度 |
| `isaacgym/` | Vendor Isaac Gym |

## 对 wiki 的映射

- [paper-mgdp-generalized-depth-perception](../../wiki/entities/paper-mgdp-generalized-depth-perception.md)
- [mgdp.md](../sites/mgdp.md)
- [mgdp_adv_sci_2026](../papers/mgdp_adv_sci_2026.md)
