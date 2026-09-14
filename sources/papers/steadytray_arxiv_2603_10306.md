# SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning（arXiv:2603.10306）

> 来源归档（ingest）

- **标题：** SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning
- **简称：** SteadyTray / ReST-RL
- **类型：** paper / humanoid / loco-manipulation / residual-rl / tray-transport
- **arXiv：** <https://arxiv.org/abs/2603.10306>
- **PDF：** <https://arxiv.org/pdf/2603.10306>
- **项目页：** <https://steadytray.github.io/> — 归档见 [`sources/sites/steadytray.md`](../sites/steadytray.md)
- **代码：** <https://github.com/AllenHuangGit/steadytray> — 归档见 [`sources/repos/steadytray.md`](../repos/steadytray.md)
- **机构：** 加州大学圣地亚哥分校（UCSD）
- **入库日期：** 2026-09-14
- **最后更新：** 2026-09-14
- **一句话说明：** ReST-RL 把行走与托盘载荷稳定解耦：预训练 base locomotion + 残差模块抵消步态扰动；四阶段课程；G1 真机零样本 sim-to-real，仿真 96.9% 变速跟踪 / 74.5% 抗扰。

## 开源状态（步骤 2.5，2026-09-14）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线 |
| 训练代码 | **已开源**（`AllenHuangGit/steadytray`） |
| IsaacLab fork | **已开源**（`AllenHuangGit/IsaacLab_SteadyTray`） |
| 预训练权重 | **已提供**（`model/model_9999.pt`） |
| MuJoCo sim2sim | **已开源** |

**结论：已开源**

## 核心论文摘录（MVP）

### 1) ReST-RL：行走与载荷稳定分层

- **ReST-RL** 显式解耦 **locomotion** 与 **payload stabilization**；base policy 负责双足稳定，**残差模块**（Residual Action / FiLM Adapter）主动抵消步态引起的末端扰动。
- 对比端到端单体策略，残差设计在步态平滑度与托盘朝向精度上显著更好。
- **对 wiki 的映射：** [paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md)、[Humanoid Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) 四阶段课程 + 特权蒸馏

- Stage 1 行走预训练 → Stage 2 端盘奖励微调 → Stage 3 残差教师（特权 robot/payload 观测）→ Stage 4 **仅蒸馏 encoder**，adapter 冻结。
- **对 wiki 的映射：** [paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md)、[ResMimic](../../wiki/entities/paper-resmimic.md)

### 3) G1 真机零样本与 SteadyTray benchmark

- 仿真：**96.9%** 变速跟踪成功率、**74.5%** 外力扰动鲁棒性；真机 **零样本 sim-to-real**，多物体与扰动下保持托盘水平。
- **对 wiki 的映射：** [paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

## 当前提炼状态

- [x] 项目页与 GitHub 核查（步骤 2.5）
- [x] 四阶段训练管线摘录
- [x] wiki 映射
