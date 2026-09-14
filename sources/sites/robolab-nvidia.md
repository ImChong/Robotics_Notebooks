# RoboLab（NVIDIA SRL 项目页）

> 来源归档

- **标题：** RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies
- **类型：** site（项目页 + Leaderboard + 交互结果面板）
- **URL：** <https://research.nvidia.com/labs/srl/projects/robolab/>
- **Leaderboard：** <https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html>
- **论文：** <https://arxiv.org/abs/2604.09860>
- **代码：** <https://github.com/NVLabs/RoboLab>
- **机构：** NVIDIA（SRL）；合作方含多伦多大学、悉尼大学
- **会议：** Robotics: Science and Systems (RSS) 2026，Sydney
- **入库日期：** 2026-09-14
- **一句话说明：** 面向**真机数据训练、零样本进仿真**的通用操纵策略评测平台：RoboLab-120 任务、三能力轴×三难度、场景/任务/agent 生成工作流、NPE 敏感性分析与 RoboArena 真机排名 **Spearman ρ=0.94**。

## 开源核查（步骤 2.5，2026-09-14）

| 资源 | 状态 | 说明 |
|------|------|------|
| 评测框架 | **已开源** | [NVLabs/RoboLab](https://github.com/NVLabs/RoboLab)（Apache-2.0）：Isaac Lab 任务库、server-client 策略接口、并行评测、结果 Dashboard |
| 任务/场景资产 | **已开源** | 仓内 `assets/objects`、`assets/scenes`、`robolab/tasks`；312+ 物体库 |
| 策略权重 | **按策略而定** | 官方 Leaderboard 标注 Closed-source / Verified 等；内置 `policies/pi0_family` 对接 OpenPI π0.5 |
| 提交上榜 | **表单申请** | Leaderboard 页「Submit Your Model」；非全自动 PR 上榜 |

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Framework | 场景生成 → 任务生成（语言指令）→ 环境生成（机器人/策略/观测动作配置） |
| Benchmark | RoboLab-120；Visual / Relational / Procedural 三轴；与 DROID 训练分布对比（仅 68.7% 物体词表重叠） |
| Sensitivity | NPE 后验估计环境参数对成功率影响；腕部相机高敏感 |
| Leaderboard | RoboLab-120 Overall；按难度/能力轴拆分；语言 specificity（Vague/Default/Specific） |
| Sim-Real | 与 RoboArena Elo **Spearman ρ=0.94** |

## Leaderboard 快照（Default 指令，2026-09-14 页上数据）

| 排名 | 策略 | SR% | Score |
|------|------|-----|-------|
| 1 | OASIS WAM | 39.0 | 53.7 |
| 2 | Cosmos3-Nano-Policy | 36.8 | 51.9 |
| 3 | Phoenix | 34.4 | 45.9 |
| 5 | **π0.5** | **28.0** | **43.4** |
| 6 | DreamZero | 25.7 | 39.8 |

## 对 wiki 的映射

- 主实体：[RoboLab](../../wiki/entities/robolab.md)
- 论文摘录：[robolab_arxiv_2604_09860.md](../papers/robolab_arxiv_2604_09860.md)
- 仓库：[robolab.md](../repos/robolab.md)
- 交叉：[RoboDojo](../../wiki/entities/robodojo.md)、[π0.5](../../wiki/entities/paper-pi05-open-world-vla.md)、[Hydra-0](../../wiki/entities/paper-hydra-0.md)、[GPT 6 Astra 评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md)
