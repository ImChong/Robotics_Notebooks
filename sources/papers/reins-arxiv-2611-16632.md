# Reins: Frontier Multimodal Models Driving Real Robot Arms on Long-Horizon Tasks, Zero-Shot

- **类型：** paper / robotics
- **中文标题：** Reins：前沿多模态模型零样本驱动真机长程双臂操作
- **论文：** [arXiv:2611.16632v1](https://arxiv.org/abs/2611.16632)（2026 年 11 月）
- **HTML 全文：** https://arxiv.org/html/2611.16632v1
- **PDF：** https://arxiv.org/pdf/2611.16632
- **作者：** Pengfei Ye、Zhiyang Dou、Yuxiang Ma、Haonan Chen、Heng Yang、Edward H. Adelson
- **机构：** 麻省理工学院 MIT CSAIL、哈佛大学
- **项目：** https://reins-robot.github.io/
- **源码：** 未查到独立可运行的官方发布入口
- **入库日期：** 2026-11-16
- **核对范围：** 已核对原始论文 HTML v1 的引言、方法、实验、对比、结论与局限，并参照作者项目页核对核心机制及口径；参考文献不逐篇复读。
- **通读：** 已完成
- **一句话说明：** 无需任务专属训练，让现成多模态模型通过三维视觉工具、带守护条件的原子动作链和并行场景旁白操作真实双机械臂。

## 原文核对摘要与 wiki 映射

### 方法与控制接口

Actor 观察图像/任务和历史，调用 find/measure 三维感知与原子动作工具；一次调用可给多步动作链，每一步检查 success guard，首个失败即停止回报状态；另一个 narrator 实例持续观测相机并把场景变化写入下一轮 Actor 上下文。可利用图像构建用于后续策略学习的仿真场景孪生，但论文的零样本真机成功率并非该孪生训练的结果。

- **对 wiki 的映射：** [Reins：前沿多模态模型零样本驱动真机长程双臂操作](../../wiki/entities/paper-reins-robot-harness.md) — 核心原理及 Mermaid 流程。

### 实验与数值边界

官方论文报告三种前沿模型的五项真机长程任务；其中 GPT-6.1 Sol 以自然语言任务提示、无任务专属代码达到 88% 的真实 episode 成功率。模型思考延迟占总任务时间 41%–78%；不能把论文中的 88% 误写为三种模型的共同平均值。

- **对 wiki 的映射：** 主实体 — 实验与评测、结论及误读警示。

### 开源、局限与工程取舍

- **开源状态（2026-11-16）：** 项目页标注 Code coming soon；未提供可运行源码入口。
- **论文边界：** 示例为两台 6-DoF AirBot Play 与所述相机/工具系统，模型和 API 服务依赖强，零样本不等于零工程接口；官方项目页明确 Code coming soon，暂无公开可复现的 Reins 运行入口。
- **对 wiki 的映射：** 主实体 — 工程实践、源码运行时序与局限。

## 原始链接

- [arXiv HTML v1](https://arxiv.org/html/2611.16632v1)
- [作者项目页](https://reins-robot.github.io/)
