# MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI

> 论文来源归档（ingest）

- **标题：** MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI
- **类型：** paper / autonomous drone racing / monocular vision + reinforcement learning
- **作者：** Stavrow A. Bahnam、Robin Ferede、Till M. Blaha、Anton E. Lang、Erin Lucassen、Quentin Missinne、Aderik E. C. Verraest、Christophe De Wagter、Guido C. H. E. de Croon
- **机构：** 代尔夫特理工大学（Delft University of Technology）
- **期刊：** Science Robotics，2026-10-14
- **DOI：** <https://doi.org/10.1126/scirobotics.aee9250>
- **arXiv：** <https://arxiv.org/abs/2601.15222>（2026-01-21）
- **PDF：** <https://arxiv.org/pdf/2601.15222>
- **官方报道：** <https://www.tudelft.nl/2026/tu-delft/de-wetenschap-achter-s-werelds-snelste-autonome-racedrone>
- **代码：** 论文与校方报道未列公开代码仓库或下载入口；公开实现状态未确认。
- **入库日期：** 2026-10-25
- **一句话说明：** MonoRace 将单目滚动快门相机、IMU 与无人机模型结合进行高速状态估计，并以 500 Hz 小型神经网络直接生成电机命令；论文报告其赢得 2025 年 A2RL 自主无人机竞速赛，在赛道上达到最高约 100 km/h。

## 核心论文摘录

### 1) 单目视觉与机体模型共同估计高速状态

- **要点：** 机载前视单目滚动快门相机通过神经网络分割赛道门框；IMU 与无人机模型共同用于估计位置和运动。系统不依赖 GPS 或外部运动捕捉。
- **对 wiki 的映射：**
  - [MonoRace 论文实体页](../../wiki/entities/paper-monorace-champion-level-drone-racing.md)
  - [State Estimation](../../wiki/concepts/state-estimation.md)

### 2) 利用赛道几何离线校准估计参数

- **要点：** 离线优化利用已知门框几何和机载飞行数据，细化状态估计参数，尤其是相机外参；该过程不依赖外部定位系统。
- **对 wiki 的映射：**
  - [MonoRace 论文实体页](../../wiki/entities/paper-monorace-champion-level-drone-racing.md)
  - [状态估计](../../wiki/concepts/state-estimation.md)

### 3) 神经网络直接输出电机命令

- **要点：** 控制网络绕过传统多层内环控制器，直接给出电机命令，在机载飞控上以 500 Hz 运行；通过强化学习训练，将无人机物理约束纳入控制行为。
- **对 wiki 的映射：**
  - [MonoRace 论文实体页](../../wiki/entities/paper-monorace-champion-level-drone-racing.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 4) 竞赛结果与鲁棒性

- **要点：** 论文报告 MonoRace 赢得 2025 年 Abu Dhabi Autonomous Drone Racing Competition（A2RL），在直接淘汰赛中超过其他 AI 队伍及三位人类世界冠军飞手；赛道最高速度约 100 km/h，并应对相机干扰和 IMU 饱和。
- **对 wiki 的映射：**
  - [MonoRace 论文实体页](../../wiki/entities/paper-monorace-champion-level-drone-racing.md)
  - [强化学习](../../wiki/methods/reinforcement-learning.md)

## 当前提炼状态

- [x] 论文摘要、期刊 DOI 与 TU Delft 官方报道已归档
- [x] 已提炼感知、状态估计、控制频率与竞赛结果
- [ ] 官方代码或数据入口待后续核实
