# RSS 2026 | 面向基于 MoE 的鲁棒四足运动：可靠 Sim-to-Real 可预测性【文献解读】

> 来源归档（blog / 微信公众号）

- **标题：** RSS 2026 | 面向基于 MoE 的鲁棒四足运动：可靠 Sim-to-Real 可预测性【文献解读】
- **类型：** blog
- **作者：** PinkRobot（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/xPABD1wBfptOzqsIbqHYMw
- **发表日期：** 2026-09-15（入库日）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch
- **论文：** Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion（RSS 2026）
- **arXiv：** <https://arxiv.org/abs/2602.00678>
- **项目页：** <https://robogauge.github.io/complete/>
- **一句话说明：** 西安交大团队：student latent encoder 上放 **MoE**（非 action 侧）+ **RoboGauge** 跨 Isaac Gym→MuJoCo 分层压力测试，用 sim-to-sim 指标在真机前选 checkpoint；Go2 未见地形与 4 m/s 高速窄步态涌现。

## 步骤 2.5（开源核查）

- **已开源（项目页 2026-09-15 核查）：**
  - 训练：<https://github.com/wty-yy/go2_rl_gym>
  - 评估：<https://github.com/wty-yy/RoboGauge>
  - 部署：<https://github.com/wty-yy/unitree_cpp_deploy>
- **硬件：** Unitree Go2 / Go2-Edu；训练 Isaac Gym，评估 MuJoCo

## 核心摘录（归纳）

### 双贡献闭环（工程迭代，非端到端联合优化）

1. **MoE student encoder：** 多 expert 先形成地形/命令条件化 latent，再由统一 actor 映射动作；避免 action 侧 MoE 导致训练发散（AC-MoE/MCP 消融）。
2. **RoboGauge：** 独立于训练引擎；Base → Multi/Level → Stress 三 pipeline；8 项指标（tracking + safety + quality + ZMP/friction margin）；几何平均防「单项满分掩盖短板」。

### 训练规模（文内）

- 8192 并行 agent；200 Hz 物理 / 50 Hz 控制；5 帧观测历史
- 七类地形：flat / wave / slope / rough slope / stairs up/down / obstacle
- Command curriculum + extreme sampling + dynamic command sampling（附录 dynamic sampling ≈ +11% RoboGauge）

### 为什么 Sim-to-Sim 能预测 Sim-to-Real

策略若在训练仿真器 $P_{\text{train}}$ 上 exploit 数值细节，换引擎 $P_{\text{eval}}$ 性能会掉；RoboGauge 测的是 **对动力学 shift 的敏感度**，作为保守部署前 proxy。

## 对 wiki 的映射

- **升格/补强：** [paper-robogauge-moe-quadruped-locomotion](../../wiki/entities/paper-robogauge-moe-quadruped-locomotion.md)（新建完整实体；与 [paper-notebook stub](../../wiki/entities/paper-notebook-toward-reliable-sim-to-real-predictability-for-m.md) 互链）
- **交叉：** [Sim2Real](../../wiki/concepts/sim2real.md)、[Domain Randomization](../../wiki/concepts/domain-randomization.md)、[Locomotion](../../wiki/tasks/locomotion.md)
