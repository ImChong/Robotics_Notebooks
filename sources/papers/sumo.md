# Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation

- **标题**: Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation
- **链接**: https://arxiv.org/abs/2604.08508
- **作者**: John Z. Zhang, Maks Sorokin, Zachary Manchester, Simon Le Cléac’h 等
- **机构**: RAI Institute, Carnegie Mellon University (CMU)
- **发表日期**: 2026-04
- **核心关注点**: 全身移动操作 (Whole-Body Loco-Manipulation)、层级控制 (Hierarchical Control)、MPC-over-RL、重物操纵

## 核心摘要

Sumo 提出了一种层级化架构，通过结合**采样模型预测控制 (Sample-based MPC)** 和**强化学习 (RL)**，使腿式机器人（Spot, G1）能够操纵比自身更重、更大的物体（如 15kg 的轮胎、大型护栏）。

### 1. 核心架构：MPC-over-RL
与传统的“高层 RL + 低层 WBC/MPC”架构不同，Sumo 采用反向层级：
- **底层 (Low-Level RL)**：预训练的通用全身控制策略（如 Relic）。负责 50Hz 的高频稳定与步态执行，将不稳定的足式动力学转化为稳定的命令空间（Command Space）。
- **高层 (High-Level MPC)**：基于采样的在线规划器（CEM 方法），运行频率 20Hz。它在底层策略的命令空间（如基座速度、末端位姿目标）中采样，而非原始力矩空间。

### 2. 技术优势
- **Policy-in-the-Loop Rollouts**：MPC 在进行前向预测时，将底层 RL 策略直接包含在模拟循环中。这极大地降低了搜索空间的维度，并防止了腿式机器人在单次射击规划中常见的发散问题。
- **零样本泛化 (Zero-shot Generalization)**：通过在部署时切换 MPC 的代价函数（Cost Function）或物体模型，即可适应新任务，无需重新训练 RL 模型。
- **高效性**：相比端到端 RL，调参计算时间减少 10 倍，且只需极简的代价项。

### 3. 实验验证
- **Spot 机器人**：成功扶起 15kg 轮胎（超过手臂 11kg 的额定载荷）、拖动大型围栏、堆叠轮胎。
- **G1 人形机器人**：在仿真中验证了架构的可移植性，完成了开门和推桌子等任务。

## 对 Wiki 的映射
- **wiki/methods/sumo.md** (新建)
- **wiki/tasks/loco-manipulation.md** (补充层级架构与重物操纵案例)
- **wiki/concepts/mpc-wbc-integration.md** (补充 MPC-over-RL 的反向层级模式)
