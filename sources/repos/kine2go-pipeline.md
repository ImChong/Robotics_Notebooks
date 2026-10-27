# nomagiclab/kine2go-pipeline

> 来源归档（ingest）

- **名称：** Kine2Go pipeline
- **类型：** repo / quadruped / motion-retargeting / imitation-learning
- **URL：** <https://github.com/nomagiclab/kine2go-pipeline>
- **项目页：** [Kine2Go](../sites/kine2go-project.md)
- **论文：** [Kine2Go（arXiv:2606.14433）](../papers/kine2go_arxiv_2606_14433.md)
- **数据集：** <https://huggingface.co/datasets/MIMUW-Robotics/kine2go>
- **许可证：** BSD-3-Clause（pipeline 代码）
- **入库日期：** 2026-10-27
- **开源结论：** 已开源；公开重定向、模仿训练与 rollout 采集代码。数据集使用单独的逐来源许可，不能以代码许可证代替数据许可。

## 一句话说明

将 AI4Animation 狗动作、Vienna Horse Data Collection（VHDC）马动作与 Solo8 四足动作重定向至 Unitree Go2，在 Genesis 中训练 PPO 模仿策略并生成数据集 rollout。

## 主要模块

| 路径 / 阶段 | 作用 |
|------|------|
| motion_retargeting/ | 通过逐帧逆运动学将源动作映射为 Go2 参考轨迹 |
| motion_imitation/、go2_genesis/ | Genesis 仿真环境、PPO 模仿策略训练与评估 |
| tools/gather_trajectories.py | 从训练策略采集并后处理 rollout |

## 运行环境（README）

- Unitree Go2，12 个驱动关节
- Genesis 0.3.10；PPO 使用 rsl-rl-lib 3.0.0
- Python ≥ 3.12
- 源数据：AI4Animation、VHDC、Solo8

## 对 wiki 的映射

- [Kine2Go 数据集实体](../../wiki/entities/paper-kine2go.md)
