# PACE — Sim-to-Real Transfer for Legged Robots

> 来源归档（仓库 README 与公开文档要点摘录，非全文镜像）

- **标题：** PACE (Precise Adaptation through Continuous Evolution)
- **类型：** repo
- **组织：** leggedrobotics（ETH Zurich, Robotic Systems Lab）
- **链接：** https://github.com/leggedrobotics/pace-sim2real
- **文档站：** https://pace.filipbjelonic.com/
- **许可：** Apache-2.0
- **入库日期：** 2026-07-15
- **一句话说明：** ETH RSL 开源的足式 sim2real 管线：chirp 数据采集 → Isaac Lab 并行仿真 + CMA-ES 关节动力学辨识 → 盲 locomotion RL 训练与零样本部署；绑定 Isaac Sim 5.0 / Isaac Lab latest。
- **沉淀到 wiki：** [wiki/entities/paper-pace-sim2real-legged-robots.md](../../wiki/entities/paper-pace-sim2real-legged-robots.md)

---

## 依赖与运行面（README 声明）

- **Isaac Lab** 独立安装（conda / uv 推荐）；本仓克隆在 IsaacLab 目录 **之外**
- **Isaac Sim 5.0 / Isaac Lab (latest)** 为开发与测试基线；< 5.0 可能缺少关节粘性摩擦等物理属性（警告可忽略但精度下降）
- Python 3.10+；`python -m pip install -e source/pace_sim2real`
- 可选 pre-commit 格式化

---

## 能力边界（作者在 README / 文档中的定位）

1. **数据采集：** `scripts/pace/data_collection.py` — 生成/收集 chirp 激励数据（示例 ANYmal D → `data/anymal_d_sim/chirp_data.pt`）
2. **参数拟合：** `scripts/pace/fit.py` — CMA-ES 估计执行器与关节参数 → `logs/pace/anymal_d_sim/`
3. **环境注册：** `import pace_sim2real.tasks` 注册 Isaac Lab 任务
4. **核心 API：** `PaceCfg`, `PaceSim2realEnvCfg`, `CMAESOptimizer`, `PaceDCMotorCfg`, `PaceDCMotor`

---

## 文档站结构（pace.filipbjelonic.com）

| 分区 | 内容 |
|------|------|
| Getting started | 前置依赖、安装、首个 PACE 环境 |
| Examples | ANYmal 公开版上的参数辨识与部署最小脚本 |
| Guides / Basics | 自建 PACE 环境分步指南 |
| Guides / Advanced | 自定义目标、参数与优化逻辑 |
| Best practices | 实践建议与常见陷阱 |
| Concepts / API / Development | 规划中 |

---

## 维护者与致谢（README）

- Maintainers：Filip Bjelonic, René Zurbrügg, Oliver Fischer（ETH RSL）
- 致谢 RSL Learning Group 及多位在研究中扩展 PACE 的协作者

---

## 与本仓库其他资料的关系

| 资料 | 关系 |
|------|------|
| [pace_sim2real_arxiv_2509_06342.md](../papers/pace_sim2real_arxiv_2509_06342.md) | 论文方法与实验叙事 |
| [pace-filipbjelonic-com.md](../sites/pace-filipbjelonic-com.md) | MkDocs 文档站溯源 |
| [wiki/entities/robotic-world-model-eth-rsl.md](../../wiki/entities/robotic-world-model-eth-rsl.md) | 同 ETH RSL + Isaac Lab 生态 |
| [wiki/entities/sage-sim2real-actuator-gap-estimator.md](../../wiki/entities/sage-sim2real-actuator-gap-estimator.md) | 另一执行器层 gap 度量/对齐工具链 |
| [wiki/methods/actuator-network.md](../../wiki/methods/actuator-network.md) | PACE 论文对比的黑盒执行器建模基线 |
