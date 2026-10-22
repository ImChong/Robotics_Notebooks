# Spotter 代码仓

> 来源归档（repo）；2026-10-22 浅克隆（`--depth 1`，提交 `0ad930d`）核查。

- **仓库：** https://github.com/zqc3117/Spotter
- **许可：** MIT（README 徽章与 LICENSE）
- **论文：** [Spotter 论文归档](../papers/spotter_arxiv_2609_36808.md)
- **项目页：** https://zqc3117.github.io/Spotter/
- **仿真器：** RoboCasa、RoboTwin 2.0
- **发布内容（README 列出）：**
  - 模型权重链接：pi0.5（`DAVIAN-Robotics/pi05-robocasa-H50`）、Cosmos Policy（`nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`）、RoboTwin 2.0 的 pi0.5（`motus-robotics/pi0.5_robotwin2`）、Qwen 截屏与裁判模型
  - 评测入口：`spotter.sh`、`robotwin_eval.sh`；RoboTwin 50 任务 × 10 episode 数据集（ModelScope）
- **状态：** 核查时仓库含可运行入口与安装说明；LFS 文件在浅克隆中未展开，权重与依赖未在本次实际运行
- **入库日期：** 2026-10-22
