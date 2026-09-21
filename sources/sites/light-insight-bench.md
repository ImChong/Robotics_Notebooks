# INSIGHT-Bench 项目页（lightorigins.github.io）

- **标题：** INSIGHT-Bench — Object-goal navigation in NVIDIA Isaac Sim
- **类型：** site
- **项目页：** <https://lightorigins.github.io/Light-INSIGHT-Bench/>
- **论文：** [arXiv:2608.30935](https://arxiv.org/abs/2608.30935)（LightNav-0）
- **代码：** <https://github.com/lightorigins/Light-INSIGHT-Bench>
- **数据集：** <https://huggingface.co/datasets/LightOriginsHQ/light-insight-bench>
- **机构：** 亮源新创（Light Origins）
- **入库日期：** 2026-09-21

## 开源核查（2026-09-21）

| 资产 | 状态 |
|------|------|
| 评测 harness + Isaac Lab runner | **已开源** `lightorigins/Light-INSIGHT-Bench`（Apache-2.0） |
| 1097 episode 评测 split + 10 个 Habitat-GS 场景 | **已发布** HF `LightOriginsHQ/light-insight-bench`（无 gate） |
| 其余 200 场景 | **需自行按源许可获取**（InteriorGS / HM3D / MP3D 等，见仓库 `guides/scenes.md`） |
| 训练侧 1683 场景 / 53090 片段 | **未随本仓库完整发布**；博客口径为 Real2Sim2Real 数据引擎产物 |

## 核心摘录

1. **诊断式评测：** 210 held-out 场景、1097 episodes；5×5（场景类 × 指令类型）矩阵把 aggregate SR 拆成「布局 vs 语言机制 vs 交互」。
2. **统一部署协议：** 前向单目 RGB 480×270、120° FOV、1.0 m 相机高度；300 action budget；室内 2 m / 室外 3 m 成功半径；成功需 stop 且目标在最终帧 FOV 内。
3. **可验证 leaderboard：** 提交为 PR + evidence pack；CI 拉取 zip、校验 sha256、`insight-bench verify` 后计分；`published` 与 `verified` 分列。

## 交叉链接

- 论文摘录：[insight_bench_lightorigins_2026](../papers/insight_bench_lightorigins_2026.md)
- 仓库：[lightorigins/Light-INSIGHT-Bench](../repos/lightorigins-light-insight-bench.md)
- 主实体：[INSIGHT-Bench](../../wiki/entities/insight-bench.md)
- 关联论文实体：[LightNav-0](../../wiki/entities/paper-lightnav-0.md)
