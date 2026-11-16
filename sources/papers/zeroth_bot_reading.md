# Zeroth Bot（官方开源仓库与文档归档）

> 来源归档（ingest · 官方 GitHub 仓库与文档溯源，无同行评审学术论文）

- **标题：** Zeroth-01 Bot (<https://github.com/zeroth-robotics/zeroth-bot>)
- **类型：** repo（Paper Notebooks `12_Hardware_Design` 收录条目，无对应 arXiv 论文）
- **深读状态：** 已改为直接对照官方仓库核对（原 `PROGRESS.md` 待深读目录无对应论文 PDF）
- **分类：** 12_Hardware_Design
- **官方代码：** <https://github.com/zeroth-robotics/zeroth-bot>
- **官方文档：** <https://docs.kscale.dev/docs/zeroth-01>
- **核对范围：** 核对仓库根目录全部可读取文件 `README.md`、`LICENSE`（MIT，Copyright 2024 Jingxiang Mo）与 `.gitmodules`（声明子模块 `kscalelabs/kos-zbot` 与 `kscalelabs/ksim-gym-zbot`）；确认离线克隆仓库内两个子模块目录均为空目录（未初始化），且该项目无同行评审论文或 arXiv 预印本。
- **通读：** 已完成
- **入库日期：** 2026-06-11（官方仓库实读核对：2026-11-14）
- **一句话说明：** Zeroth-01 Bot 是由开源社区与 K-Scale 生态维护的低成本（BOM 起价 350 美元）3D 打印全开源端到端人形机器人平台，由顶层伞仓 `zeroth-bot` 串联机械/文档入口、`kos-zbot` 操作系统与硬件抽象层（Feetech 舵机与 IMU 驱动）以及 `ksim-gym-zbot` GPU 强化学习与 SysID Sim2Real 训练管线。

## 核心摘录（一手仓库文档核对）

- **项目定位与状态**：Public Beta 阶段，已面向基础双足行走、视觉与语音开放，官方提示存在破坏性变更预期（计划 2025 年 6 月发布稳定版 V1.0）。
- **模块化三层架构**：硬件 3D 打印结构（BOM 350 美元起）+ `kos-zbot`（`kos_zbot` Python API、Feetech 舵机/IMU 驱动、低延迟实时控制与 CLI 标定诊断工具）+ `ksim-gym-zbot`（MJCF/URDF 模型、SysID 标定执行器与 GPU 强化学习全身控制导出至 KOS）。
- **资料边界声明**：项目一手依据为 GitHub 官方仓库与 `docs.kscale.dev` 在线指南，目前没有同行评审学术论文或定量对比基准报告。

## 对 wiki 的映射

- [paper-notebook-zeroth-bot-https-github-com-zeroth-robotics-zero](../../wiki/entities/paper-notebook-zeroth-bot-https-github-com-zeroth-robotics-zero.md)

## 参考来源（原始）

- 官方仓库：<https://github.com/zeroth-robotics/zeroth-bot>
- 官方文档：<https://docs.kscale.dev/docs/zeroth-01>
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
