# AGILE: A Comprehensive Workflow for Humanoid Loco-Manipulation Learning（arXiv:2603.20147）

> 来源归档（ingest）

- **标题：** AGILE: A Comprehensive Workflow for Humanoid Loco-Manipulation Learning
- **缩写 / 框架：** **AGILE**（**A** **G**eneric **I**saac-**L**ab based **E**ngine）
- **类型：** paper / humanoid / loco-manipulation / reinforcement-learning / sim2real / workflow / isaac-lab
- **arXiv：** <https://arxiv.org/abs/2603.20147>（Submitted 2026-03-20；HTML：<https://arxiv.org/html/2603.20147v1>；PDF：<https://arxiv.org/pdf/2603.20147>）
- **代码：** <https://github.com/nvidia-isaac/WBC-AGILE>（已开源；归档见 [`sources/repos/wbc_agile.md`](../repos/wbc_agile.md)）
- **文档 / 项目站：** <https://nvidia-isaac.github.io/WBC-AGILE/>（归档见 [`sources/sites/wbc-agile-docs.md`](../sites/wbc-agile-docs.md)）
- **作者：** Huihua Zhao\*、Rafael Cathomen\*、Lionel Gulich、Wei Liu、Efe Arda Ongan、Michael Lin、Shalin Jain、Soha Pouya、Yan Chang（\* equal contribution）
- **机构：** 英伟达（NVIDIA）
- **入库日期：** 2026-08-07
- **一句话说明：** NVIDIA 开源的 Isaac Lab + RSL-RL 人形 RL **全生命周期工作流**：Prepare → Train → Evaluate → Deploy；用配置描述符统一 Sim2Sim / 真机 I/O，并在 Unitree G1 与 Booster T1 上验证 locomotion / stand-up / imitation / loco-manipulation。

## 开源状态（步骤 2.5）

- **项目站核查（2026-08-07）：** [nvidia-isaac.github.io/WBC-AGILE](https://nvidia-isaac.github.io/WBC-AGILE/) 明确指向 GitHub 仓库，列出任务 ID（`Velocity-*` / `StandUp-*` / `G1-PickPlace-*` / `Tracking-Flat-G1-v0` 等）、`scripts/train.py` / `scripts/eval.py` Quick Start，以及 Sim-to-MuJoCo / OSMO 远程训练能力。
- **仓库核查（2026-08-07）：** [nvidia-isaac/WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) README 要求 **Isaac Lab v2.3.2 + Isaac Sim 5.1**；提供 `scripts/setup/install_deps_local.sh`、`scripts/train.py`、`scripts/eval.py`、多机器人 GIF 演示与文档站；许可为 **Apache-2.0**（`agile/algorithms/rsl_rl/` 子集为 BSD 3-Clause，源自 ETH RSL-RL）。
- **结论：** **已开源**（训练 / 评测 / 任务配置 / 文档齐全）。论文注明真机 sim-to-real 驱动管线「将另行发布」——部署层以 YAML I/O 描述符 + MuJoCo Sim2Sim 为主入口；硬件驱动集成需对照文档与后续 release。

## 摘录 1：问题与主张（§1）

- **痛点（Workflow Gap）：** 人形 RL 失败常来自环境/奖励配置错误、随机 rollout 评测掩盖关节限位与高频作动等硬件关键行为，而非仿真吞吐或算法新颖性。
- **痛点（Transfer Gap）：** 导出策略时关节顺序、观测历史缓冲、动作缩放需手工对齐；无统一 I/O 合同则 Sim2Sim（MuJoCo）与真机部署易引入静默 bug。
- **主张：** AGILE 把人形 RL 开发做成 **可重复工程生命周期**：交互式环境核验 → 可复现训练（含可开关算法增强）→ 确定性场景 + 随机 rollout 统一评测（含运动质量诊断）→ **描述符驱动** 策略导出与部署。
- **验证：** 五类技能（速度跟踪、高度可控 locomotion、stand-up、运动模仿、loco-manipulation / VLA）× 两平台（Unitree G1、Booster T1）；预训练 checkpoint 随仓库发布。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-agile-humanoid-loco-manipulation.md`](../../wiki/entities/paper-agile-humanoid-loco-manipulation.md)；与 [Isaac Lab](../../wiki/entities/isaac-lab.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[Unitree G1](../../wiki/entities/unitree-g1.md) 互链。

## 摘录 2：四阶段系统（§3）

| 阶段 | 要点 |
|------|------|
| **Prepare** | Joint Position / Object Manipulation / Reward Visualizer 三类 GUI 插件；训练前分钟级抓住关节符号、碰撞与奖励项错误 |
| **Train** | 统一入口 + git snapshot / YAML dump / W&B / Docker；scaled-dict 超参扫描；可开关模块：L2C2、在线奖励归一化、value-bootstrapped terminations、virtual harness、对称增强、速度剖面（EMA/梯形/线性）、状态缓存、teacher–student 蒸馏 |
| **Evaluate** | 确定性脚本命令场景 + 随机 rollout；Isaac Lab 与 MuJoCo 共用指标（RMS 加速度/jerk、限位违例、高频能量比）；导出交互 HTML 报告 |
| **Deploy** | TorchScript + 自包含 YAML I/O 描述符（关节名、观测顺序、历史缓冲、动作缩放）；同一推理逻辑切换状态提供者即可 Sim2Sim / 真机 |

**对 wiki 的映射：** 实体页画 Prepare→Deploy 流程图与源码运行时序图；强调「工作流层」而非新算法。

## 摘录 3：案例与消融（§4–§5 / Table 2–3）

| 任务 | 机器人 | 训练量级（单卡 L40） | 读点 |
|------|--------|----------------------|------|
| Velocity | G1 / T1 | ~10 h | 同 MDP 模板跨机；对称增强 + virtual harness |
| Velocity + Height | G1 | ~10 h | 下肢 RL + 上身梯形随机化 → 解耦 WBC；MuJoCo 确定性扫参报告跟踪误差 |
| Stand-up | G1 / T1 | ~15–25 h | 奖励归一化 + fallen-pose 状态缓存 |
| Motion imitation | G1 | ~6 h | BeyondMimic 式；额外 DR + L2C2 才 Sim2Real |
| Pick & Place / VLA | G1 | ~10 h（上身 RL） | 冻结下肢；RL 专家采 100 条 → GR00T N1.5 微调；闭环仿真 **90%/100** |

- **消融（5 seeds）：** 奖励归一化抗量级漂移；L2C2 降 jerk/限位/高频能量；value-bootstrapped terminations 降种子方差；virtual harness 加速早期收敛；对称增强主要改善步态对称性。
- **局限：** 目前两平台；依赖 Isaac Lab API；任务以本体感知为主，感知驱动操作与跑步/爬楼等高动态尚未纳入；真机定量跟踪依赖外部动捕未做，定量指标走 MuJoCo 管线。

**对 wiki 的映射：** 「结论」节写清工作流贡献 vs 算法贡献；工程实践摘录附录 Best Practices 要点。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-agile-humanoid-loco-manipulation.md`**（含流程总览 + 源码运行时序图 + 结论）。
- 新建 **`sources/repos/wbc_agile.md`**、**`sources/sites/wbc-agile-docs.md`**。
- 交叉更新：[`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)、[`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/entities/unitree-g1.md`](../../wiki/entities/unitree-g1.md)、[`wiki/methods/beyondmimic.md`](../../wiki/methods/beyondmimic.md)。
