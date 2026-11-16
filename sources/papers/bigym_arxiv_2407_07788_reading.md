# BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark（原文通读归档）

> 来源归档（arXiv v2 原文全文通读；不改动共享摘要 `sources/papers/humanoid_pnb_bigym.md`）

- **标题：** BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2407.07788>（v2，17 页，标注 CoRL 2024）
- **项目主页：** <https://chernyadev.github.io/bigym/>
- **官方代码：** <https://github.com/chernyadev/bigym>
- **核对范围：** 全文通读论文文本 §1–§5（问题定义、40 个移动双手家居任务、VR 遥操作与 500 Hz 演示录制/回放机制、Gymnasium 接口、IL 与 demo-driven RL 基线评测）、Table 1–2（基准对比与 40 任务逐项成功率）、Fig. 2–5 及附录 A–C（观测/动作维度、成功判据、超参、无头服务器 400–1400 FPS 性能特征）。
- **通读：** 已完成
- **入库日期：** 2026-11-14

## 核心信息摘录

- **平台与形态**：基于 MuJoCo 搭建 Unitree H1 + 双 Robotiq 2F-85 平行夹爪移动操作仿真环境，支持 3 路 RGB-D 相机（头部、左腕、右腕，默认 84×84）与本体感受状态。
- **双模式控制**：`bi-manual` 浮动底座模式（16 维动作、64 维本体感受；下半身由经典全向底座控制器代理 `X, Y, RZ`）与 `whole-body` 全身模式（23 维动作、76 维本体感受）。
- **任务与演示**：共 40 个家居任务（Reach 3、Tabletop 5、Dishwasher 13、Kitchen Countertops 19），任务跨度 1000–7000 步；每任务 50 条 VR 遥操作人类演示，以 500 Hz 物理频率仅存动作信号，回放时生成观测并可下采样至 20–500 Hz。
- **评测结果（Table 2，40 任务均值，50 eval episodes）**：ACT 46.3 ± 1.4%、Diffusion Policy 20.8 ± 0.8%、BC 18.0 ± 1.1%、DrQ-v2 13.9 ± 0.2%、AWAC 13.0 ± 1.2%、IQL 12.4 ± 0.0%、CQN 10.5 ± 0.4%；长时程与高精度任务（如 `stack blocks`、`dishwasher unload cups long`）所有基线均为 0%。
- **核对注意点**：正文称 `put cups` 全员失败而 Table 2 列出 ACT 为 30.0 ± 7.2%；`±` 统计口径与随机种子数未在正文详述。

## 对 wiki 的映射

- [paper-notebook-bigym-a-demo-driven-mobile-bi-manual-manipulatio](../../wiki/entities/paper-notebook-bigym-a-demo-driven-mobile-bi-manual-manipulatio.md)
