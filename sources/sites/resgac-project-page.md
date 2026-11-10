# ResGAC 项目页（resgac.github.io）

> 来源归档（ingest）

- **项目页：** <https://resgac.github.io/ResGAC-website/>
- **补充详情：** <https://resgac.github.io/ResGAC-website/details.html>
- **论文：** [arXiv:2610.09479](https://arxiv.org/abs/2610.09479)
- **核查日期：** 2026-11-10
- **公开页面源码：** <https://github.com/ResGAC/ResGAC-website>（静态网站，不是训练/推理/控制代码）
- **控制器代码：** 官方项目页面显示 `Code (released on acceptance)`，README 说明双盲评审期间暂不公开代码仓库；截至核对日未发现作者可运行代码入口，不能将页面仓库误判为 ResGAC 开源实现。
- **数据：** 项目页面未列可直接下载的训练数据集（截至 2026-11-10）。

## 核对范围和开放状态

- 已核对官方 GitHub Pages 仓库的 README.md、index.html、details.html：公开的是论文链接、实验视频、训练实现参数说明与静态网站素材。
- details.html 说明 25 个奖励项、actor/critic 观测预处理和噪声、动作缩放、课程、训练与对照设置，但不能当成可执行复现实验源码。
- 官方控制代码发布后须另核验真实仓库、版本、许可证与运行入口，才能依据源码补 Mermaid 运行时序图。

## 页面信息归档

项目页展示站立轨迹跟踪、peg-in-hole 和行走时手部目标保持。补充页描述 Isaac Sim / Holosoma 与 FastSAC、4096 个并行环境、单张 L40S 约 16 小时训练、50 Hz 策略，以及部署中的 Pinocchio 运动学计算。训练课程由固定站立推进至站立轨迹、固定速度行走和行走轨迹。

这些设置可帮助理解作者的训练规模与部署结构，但在无代码、配置及数据仓库的情况下，不能称为完整可复现实现。

## Wiki 映射

- 论文实体：wiki/entities/paper-resgac-precise-se3-end-effector-tracking.md
- 论文归档：sources/papers/precise_se3_end_effector_tracking_resgac_arxiv_2610_09479.md
