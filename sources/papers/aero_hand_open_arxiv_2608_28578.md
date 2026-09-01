# Aero Hand Open（腱驱手 MuJoCo Sim2Real）

> 来源归档（ingest）

- **标题：** Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.28578>
- **项目页：** <https://tetheria.github.io/aero-hand-open/>
- **代码：** <https://github.com/TetherIA/aero-hand-open>
- **机构：** TetherIA / Chestnut Robotics（论文注：公司已更名为 Chestnut Robotics）
- **入库日期：** 2026-09-01
- **一句话说明：** $314 / 374 g / 7 电机 16 关节腱驱拟人手；发布 MuJoCo 缆绳传动模型、双向辨识作动映射与 RL 包，策略可仿真训练后零样本真机部署（含 in-hand 立方体旋转）。

## 核心摘录（策展）

### 1) 硬件：低成本腱驱全 GRASP taxonomy

- **摘录要点：** 16 关节、7 Feetech 伺服、ESP32-S3；唯一本体感知为电机编码器。覆盖 GRASP 33 类抓取；指尖约 12 N，全开合 ~1.2 Hz；>40 万次循环耐久。
- **对 wiki 的映射：**
  - [Aero Hand Open](../../wiki/entities/paper-aero-hand-open.md) — 硬件表。
  - [paper-notebook-aero-hand-open](../../wiki/entities/paper-notebook-aero-hand-open.md) — 由占位升格。

### 2) MuJoCo 缆绳级仿真

- **摘录要点：** 20 条 spatial tendon + 7 作动器；CAD 对齐 wrap 几何；PIP–DIP 与拇指 MCP–IP 等式约束。GPU 并行训练用简化碰撞 primitive + 10 ms 步长。
- **对 wiki 的映射：**
  - [Aero Hand Open](../../wiki/entities/paper-aero-hand-open.md) — 仿真节。

### 3) 作动映射与系统辨识

- **摘录要点：** 线性 joint→cable→motor 链；拇指三通道耦合；域随机化 + 双向通道验证支撑 **零样本** sim2real。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md) — 作动器级迁移先例。

### 4) RL 管线

- **摘录要点：** `sim_rl/mujoco_playground/` 训练；观测/动作仅限硬件可得信号；演示 in-hand cube rotation 零微调部署。
- **对 wiki 的映射：**
  - [aero-hand-open 仓库](../repos/aero-hand-open.md)

### 5) 开源状态（截至 2026-09-01，项目页核查）

- **摘录要点：** **已开源**：CAD/STEP、BOM、固件、Python SDK、ROS2 Humble、`sim_rl/simulation/` MuJoCo 模型、`sim_rl/mujoco_playground/` RL。设计文件 CC BY-NC-SA（非商业）；软件 Apache-2.0。
- **对 wiki 的映射：**
  - [aero-hand-open 项目页](../sites/aero-hand-open.md)
  - [aero-hand-open 仓库](../repos/aero-hand-open.md)

## 当前提炼状态

- [x] arXiv / 项目页 / GitHub 已交叉核查
- [x] 取代仅 PROGRESS 占位的 source 叙事
- [x] wiki 映射：`wiki/entities/paper-aero-hand-open.md`
