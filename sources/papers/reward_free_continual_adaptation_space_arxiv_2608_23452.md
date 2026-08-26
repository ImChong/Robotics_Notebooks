# Reward-Free Continual Adaptation for Resilient Space Robots

> 来源归档（ingest）

- **标题：** Reward-Free Continual Adaptation for Resilient Space Robots
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.23452>
  - <https://AndrejOrsula.github.io/space_robotics_bench>
- **代码：** <https://github.com/AndrejOrsula/space_robotics_bench>
- **机构：** 卢森堡大学（University of Luxembourg）
- **入库日期：** 2026-08-26
- **一句话说明：** 在无法计算部署期奖励的太空场景，预训练潜状态世界模型后冻结编码器与奖励头，只靠无监督 rollout 更新转移动态，再用想象轨迹训练策略，以适应严重硬件退化。

## 核心摘录（MVP）

### 1) 太空持续 RL 的奖励不可观测

- **摘录要点：** 轮组/推进器/执行器退化需要在线适应，但轨道与行星表面缺少外部跟踪，复杂奖励（如松散表土开挖体积）在真机不可算。持续 RL 默认依赖部署期奖励，成为在轨学习瓶颈。
- **对 wiki 的映射：**
  - [本实体页](../../wiki/entities/paper-reward-free-continual-adaptation-space.md)
  - [Space Mining](../../wiki/entities/paper-space-mining-with-robotics.md) — 地外自主与验证基础设施。

### 2) 冻结奖励结构、只校准动态

- **摘录要点：** 基于 DreamerV3 RSSM：预训练联合学编码器、转移、奖励/终止头与 actor-critic；部署后冻结 encoder/decoder/奖励头（及终止头），只更新 sequence model 与 prior 动力学（KL），学习率降一个数量级，动作加 \(\mathcal{N}(0,0.02)\) 探索噪声。策略完全在更新后世界模型的想象轨迹上重训（训练比 2048 updates/step）。
- **对 wiki 的映射：**
  - [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md) — RSSM 想象 RL。
  - [space_robotics_bench](../repos/space_robotics_bench.md) — Isaac Lab 太空任务套件。

### 3) 三域仿真故障 + 60 分钟适应窗

- **摘录要点：** 行星穿越（锁死右前轮转向+驱动，25 Hz）、轨道导航（三共位偏轴推进器全失效，10 Hz）、螺丝装配（法兰 15° 轴向弯曲，50 Hz）。预训练 20M step / 512 并行；适应阶段 **单环境、60 分钟**（90k / 36k / 180k step）。零样本因动态偏移灾难性失败；有特权奖励的适应接近从头重训上界；无奖励适应有初期恢复，但后期波动与衰减（潜表示漂移）。
- **对 wiki 的映射：**
  - [本实体页](../../wiki/entities/paper-reward-free-continual-adaptation-space.md) — 评测读法。

### 4) 开源状态（截至 2026-08-26）

- **摘录要点：** **已开源**。论文指向 `AndrejOrsula/space_robotics_bench`（Apache-2.0/MIT 双许可，Gymnasium + ROS 2，含 `scripts/dreamerv3.yaml`）。方法是在该 Bench 上的 DreamerV3 适应配方，不是独立算法仓。
- **对 wiki 的映射：**
  - [仓库归档](../repos/space_robotics_bench.md)

## 当前提炼状态

- [x] arXiv HTML + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-reward-free-continual-adaptation-space.md` 新建
