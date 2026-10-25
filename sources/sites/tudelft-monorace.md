# TU Delft：世界最快自主竞速无人机背后的科学

> 来源归档（official university news）

- **标题：** De wetenschap achter 's werelds snelste autonome racedrone
- **类型：** site / university news
- **机构：** 代尔夫特理工大学（Delft University of Technology）
- **链接：** <https://www.tudelft.nl/2026/tu-delft/de-wetenschap-achter-s-werelds-snelste-autonome-racedrone>
- **论文：** [MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI](https://arxiv.org/abs/2601.15222)
- **期刊 DOI：** <https://doi.org/10.1126/scirobotics.aee9250>
- **报道日期：** 2026-10-16
- **入库日期：** 2026-10-25
- **代码：** 本报道未列公开代码仓库或下载链接；公开实现状态未确认。
- **一句话说明：** TU Delft 介绍 MonoRace 如何以单目相机和 IMU 在无 GPS、无外部运动捕捉环境下进行高速无人机竞速，并解释 500 Hz 神经网络直接控制电机的设计。

## 页面要点

- **感知与估计：** 神经网络从相机画面识别赛道门框，IMU 与无人机模型用于估计位置和运动；已知门框几何还用于离线细化估计参数。
- **控制：** 小型机载神经网络直接生成电机命令，以 500 Hz 运行，取代控制链中多层传统内环。
- **实测：** 研究团队称系统在 2025 年 A2RL 赛事的直接淘汰赛中击败其他 AI 队伍及三位人类世界冠军飞手；速度最高约 100 km/h。
- **工程鲁棒性：** 论文展示系统应对相机干扰与 IMU 饱和的能力。报道未提供可下载代码、权重或数据集入口。

## 对 wiki 的映射

- [MonoRace 论文实体页](../../wiki/entities/paper-monorace-champion-level-drone-racing.md)
- [论文摘录](../papers/monorace_arxiv_2601_15222.md)
