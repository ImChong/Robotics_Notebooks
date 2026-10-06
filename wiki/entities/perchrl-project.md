---
type: entity
tags: [project, quadrotor, agile-perching, star-group]
status: complete
updated: 2026-10-06
summary: "STAR Group 的 PerchRL 项目入口，展示 CoRL 2026 在投的倾斜移动平台视觉栖停项目，并关联论文与两段实机演示；代码尚未公开。"
related:
  - ./paper-perchrl-2606-03441.md
sources:
  - ../../sources/sites/robotics_star_perchrl.md
---

# PerchRL 官方项目页

这是 STAR Group 对 PerchRL 项目的公开介绍节点，与 [论文详情](./paper-perchrl-2606-03441.md) 分开：此页记录项目门户展示的身份、对外材料和成熟度；方法、消融、实验数据及限制以论文节点为主，避免把同一份摘要复制成第二篇“论文”。

## 英文缩写速查

| 缩写 | 全称 | 本项目中的含义 |
|---|---|---|
| RL | Reinforcement Learning | 感知与飞行控制策略的学习框架 |
| FOV | Field of View | 机载相机的有限视场 |
| CoRL | Conference on Robot Learning | 页面标注的投稿会议 |

## 项目身份

| 字段 | 内容 |
|---|---|
| 项目名称 | PerchRL: Vision-Based Agile Perching |
| 研究组 | Smart Autonomous Robotics Group（STAR Group） |
| 页面状态 | Submitted to CoRL 2026（官网项目页） |
| 官方入口 | [STAR Group Projects](https://robotics-star.com/projects.html) |
| 论文入口 | [arXiv:2606.03441v3](https://arxiv.org/abs/2606.03441v3) |
| 官方代码 | 项目页未链接代码；论文表示 source code will be released，当前按未公开记录 |
| 视频 | [Video 1](https://www.bilibili.com/video/BV1t2j863ERi) · [Video 2](https://www.bilibili.com/video/BV12LVm6yExG) |

项目页对外概括：PerchRL 用 state-based pre-training 接 vision-based fine-tuning；随机化平台轨迹和 temporal augmentation 面向不同运动泛化；visibility-aware state augmentation 与 active-perception rewards 面向有限视场中的间歇视觉丢失。官网还声称系统在多种四旋翼平台上完成了仿真与真机实时栖停。细节和量化结果见独立论文节点。

## 页面材料及解读

- STAR Group 项目列表页将 PerchRL 单列为研究项目，并标注 CoRL 2026 投稿状态。
- STAR Group Publications 页同列论文标题与作者，并公开两条视频链接；这些是项目演示入口，不是代码或可复现实验包。
- arXiv v3 作者名单含 Yitao Zeng；项目/出版物列表的公开作者行可能滞后于 v3。以指定论文版本元数据为论文作者来源。
- 该页面没有给出独立的项目仓库、权重下载、安装步骤、训练配置或数据集链接；因此本项目节点不提供臆造的运行命令。

## 关联页面

- [论文与方法、实验、限制](./paper-perchrl-2606-03441.md)
- [官方项目页来源归档](../../sources/sites/robotics_star_perchrl.md)
- [arXiv 论文来源归档](../../sources/papers/perchrl_arxiv_2606_03441_v3.md)


## 参考来源

- [STAR Group 项目页来源归档](../../sources/sites/robotics_star_perchrl.md) — 项目介绍与公开媒体入口
- [arXiv v3 论文来源归档](../../sources/papers/perchrl_arxiv_2606_03441_v3.md) — 论文技术细节和实验
