# lift-humanoid.github.io（LIFT 项目页）

> 来源归档（ingest）

- **标题：** LIFT — Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Controls
- **类型：** site / project-page
- **官方入口：** <https://lift-humanoid.github.io/>
- **入库日期：** 2026-05-17
- **一句话说明：** 论文配套站点：汇总 **三阶段框架图**、**MuJoCo Playground 预训练** 与 **Brax 上 Booster T1 / Unitree G1 微调** 视频叙事、**零样本室外/多样地面** 部署片段，以及 **LAFAN1 + BeyondMimic 风格全身跟踪** 的初步扩展结果；外链 **YouTube** 真机微调演示与 **arXiv / GitHub**。

## 页面公开信息（检索自 2026-05-17）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://lift-humanoid.github.io/> |
| 论文 abs | <https://arxiv.org/abs/2601.21363> |
| 代码仓库 | <https://github.com/bigai-ai/LIFT-humanoid> |
| 真机微调视频（页面嵌入） | <https://youtu.be/0a-9mql4Hbs> |

## 实验叙事摘录（便于 wiki 溯源）

- **真机微调（Booster T1）：** 先在 **T1LowDimJoystickFlatTerrain** 预训练并削弱部分能量约束、平地任务；站点说明该策略 **跨仿真器（MuJoCo → Brax）可迁**，但 **零样本 sim-to-real 失败**，再以此初始化做 **分钟级** 实机数据采集与微调，步态随数据量 **80–590 s** 逐步稳定（以页面与视频为准）。
- **零样本部署展示：** 报告 **T1LowDimJoystickRough** 预训练策略在 **草地 / 上坡 / 下坡 / 泥地** 等未见面上的户外片段（与论文 outdoor zero-shot 叙事一致）。
- **Brax 交互 Demo：** 页面提供 **拖拽视角 + 点击固定机位** 的 Web 演示入口，对比微调前后 **目标前进速度跟踪** 与 **侧向漂移**。
- **BeyondMimic / LAFAN1：** 在 MuJoCo Playground 用 **JAX 复现 BeyondMimic 观测与奖励结构**，以 **LAFAN1** 片段展示全身跟踪 **初步结果**（站点定位为 broader applicability 证据，非主文核心 claim）。

## 对 wiki 的映射

- [`wiki/entities/lift-humanoid.md`](../../wiki/entities/lift-humanoid.md)
- [`sources/papers/lift_humanoid_arxiv_2601_21363.md`](../papers/lift_humanoid_arxiv_2601_21363.md)
- [`sources/repos/bigai-lift-humanoid.md`](../repos/bigai-lift-humanoid.md)
