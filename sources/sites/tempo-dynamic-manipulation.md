# TEMPO 动态操作项目页

> 来源归档

- **标题：** TEMPO: Learning Temporal Context for Dynamic Robot Manipulation
- **类型：** site
- **URL：** <https://tempo-robot.github.io/>
- **论文：** <https://arxiv.org/abs/2609.16864>
- **代码：** <https://github.com/tempo-robot/TEMPO>（模型实现）/ <https://github.com/tempo-robot/TempoBench>（基准数据集）
- **机构：** University of California, Irvine
- **入库日期：** 2026-10-10
- **通读：** 已完成
- **核对范围：** 已通读项目主页完整快照 `sites/tempo_dyn.html`（L1–868），核对发表荣誉（CoRL 2026 Oral, top 1%）、作者归属及仓库双链接（`TEMPO` 与 `TempoBench`）。
- **对应实体：** [TEMPO 动态操作](../../wiki/entities/paper-tempo-dynamic-manipulation.md)

## 开源核查（2026-11-14 复核）

项目页 `sites/tempo_dyn.html` 明确标注论文被 **CoRL 2026（Oral, top 1%）** 接收，并在主页导航区同时给出两个官方 GitHub 仓库链接：
1. <https://github.com/tempo-robot/TEMPO>：模型实现、SAM 2 特征预计算、PyTorch 微调与 `serve_policy.py` 推理代码（训练后权重与 YAM 真机部署仍在 README TODO 中）。
2. <https://github.com/tempo-robot/TempoBench>：已实际公开 **TEMPO-Bench** 数据集（LeRobot v2.1 格式，257 episodes / 182 已标注 / 97,265 总帧 / 55,482 定位帧，覆盖 `ball_drop`、`dynamic_pour`、`dynamic_handover`、`spartan`、`catch_ball`，附 `tempobench` 读取器与 MuJoCo 渲染脚本；`LICENSE` 仍为占位文本）。

- [论文来源归档](../papers/tempo_arxiv_2609_16864.md)
- [官方代码来源归档](../repos/tempo-robot-tempo.md)
