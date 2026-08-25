# G1 软地面参考引导起身（Compliant-Surface Stand-Up）

> 来源归档（ingest）

- **标题：** Demonstration-Guided Humanoid Stand-Up on an Emulated Deformable Surface
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.20852>
- **代码：** <https://github.com/andireposit/Stand-Up-Motion-on-Compliant-Surface-for-Humanoid>
- **权重/数据：** <https://github.com/andireposit/Stand-Up-Motion-on-Compliant-Surface-for-Humanoid/tree/main/model>
- **机构：** 印度理工学院坎普尔分校（IIT Kanpur），机械工程系
- **入库日期：** 2026-08-25
- **一句话说明：** 用硬地采集并重定向的人形起身演示作参考，PPO 残差关节控制 + 显式恢复奖励；两阶段训练（硬地→MuJoCo solref/solimp 软接触）使 29-DoF Unitree G1 在仿真软地面完成起身。

## 核心摘录（MVP）

### 1) 问题：合规地面的延迟支撑力

- **摘录要点：** 软地面需显著穿透才产生足够接触力，改变姿态与平衡；现有起身 RL/模仿工作多假设刚性地或仅轻度 DR，未专门研究「硬地演示→软地执行」的参考适配。
- **对 wiki 的映射：**
  - [G1 Compliant-Surface Stand-Up](../../wiki/entities/paper-g1-compliant-surface-standup.md) — 问题设定。
  - [Balance Recovery](../../wiki/tasks/balance-recovery.md) — 起身子技能谱系。

### 2) 参考引导残差 PPO + 双阶段课程

- **摘录要点：** BONES-SEED 经 GMR/PyRoki 重定向到 G1；策略 60 Hz 输出 29 维残差修正参考关节位（scale 0.25）+ PD 跟踪。奖励 = 参考跟踪（位/速/根姿态）+ **最终恢复**（骨盆高度、躯干竖直、终态站姿）+ 正则；消融显示仅跟踪参考不足以起身。阶段一硬地；阶段二降低 solref/solimp（**solref=(0.1,1), solimp=(0.0,0.95,0.02)**）并调参+重置噪声课程。
- **对 wiki 的映射：**
  - [G1 Compliant-Surface Stand-Up](../../wiki/entities/paper-g1-compliant-surface-standup.md) — 方法与奖励表。
  - [HoST](../../wiki/entities/paper-host-humanoid-standingup.md) — 无参考纯 RL 起身对照。

### 3) 仿真结果与发布物

- **摘录要点：** 代表策略终态骨盆高 **0.792 m**（目标 0.794 m）、竖直度 **0.991**、最大接触穿透约 **40 mm**；两条起身序列均在硬/软地达标。仓库含 `eval.py`、MuJoCo `g1.xml`、参考 CSV、**已训 policy zip** 与 vecnormalize。
- **对 wiki 的映射：**
  - [stand-up-compliant-surface 仓库](../../sources/repos/stand_up_compliant_surface_humanoid.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 4) 开源状态（截至 2026-08-25）

- **摘录要点：** **已开源（部分）**：评测脚本、MuJoCo 模型、参考动作、**训练好的软地策略权重**；**未发布**完整训练代码与多轨迹训练管线（README 侧重 `eval.py` 复现）。
- **对 wiki 的映射：**
  - [G1 Compliant-Surface Stand-Up](../../wiki/entities/paper-g1-compliant-surface-standup.md) — 工程实践表。

## 当前提炼状态

- [x] arXiv + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-g1-compliant-surface-standup.md` 新建
