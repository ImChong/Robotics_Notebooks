# FAR-hands/play_policy

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/FAR-hands/play_policy>
- **项目页：** <https://play-policy.github.io/>
- **论文：** [arXiv:2610.22340](https://arxiv.org/abs/2610.22340)
- **源码状态：** 已公开仿真实验管线、checkpoint 与数据（核验日期：2026-10-19）
- **知识节点：** [DSP](../../wiki/entities/paper-dsp-supervised-play.md)

## README 入口

- `quickstart/`：checkpoint 下载、MPC 单步验证和 Isaac Lab 闭环评测。
- `docs/DATASET.md`：人类 teleoperation 与 5 小时自主 play 数据说明。
- `scripts/pipeline.sh`、`docs/PIPELINE.md`：从 teleoperation、play policy、自主采集、world model 到 MPC 的仿真流水线。
- 环境依赖包括 Isaac Sim 5.1、Isaac Lab 2.3.2；单次完整 128-trial 评测 README 估计约 45 分钟且使用 RTX 5090。

## 关联资料

- [项目页归档](../sites/dsp-play-policy.md)
- [论文摘录](../papers/dsp_arxiv_2610_22340.md)
