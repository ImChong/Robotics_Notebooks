---
type: entity
tags:
  - paper
  - wam
  - tactile
  - manipulation
  - contact-rich
status: complete
updated: 2026-09-18
arxiv: "2609.20761"

related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ./paper-lawam.md
  - ../tasks/manipulation.md
  - ../overview/contact-wm-10-papers-technology-map.md
sources:
  - ../../sources/papers/agile_wam_arxiv_2609_20761.md
  - ../../sources/sites/agile-wam-taro.md
  - ../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md
summary: "Agile-WAM（arXiv:2609.20761）：轻量触觉 WAM：共享 latent + vision-tactile-to-action flow matching；9 仿真 + 5 真机；11.9 ms 推理、真机 SR 相对 +29.4%。"
---

# Agile-WAM（arXiv:2609.20761）

**Agile-WAM**（*An Agile Tactile World Action Model for Contact-Rich Robot Control*，[arXiv:2609.20761](https://arxiv.org/abs/2609.20761)，[项目页](https://hanchuzhou.github.io/TARO_project_page/)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)（策展档位：**深读**）。

## 一句话定义

**轻量触觉 WAM：共享 latent + vision-tactile-to-action flow matching；9 仿真 + 5 真机；11.9 ms 推理、真机 SR 相对 +29.4%。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World-Action Model | 联合预测未来观测与动作的策略 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| SR | Success Rate | 任务成功率 |
| IL | Imitation Learning | 模仿学习 |
| GS | Gaussian Splatting | 高斯溅射三维表示 |

## 为什么重要

- 公众号将本文归入「接触时视觉之外还需预测什么」专题；深读档位。
- **加州大学戴维斯分校（UC Davis）；Analog Devices**；开源结论：**待发布**（步骤 2.5，2026-09-18）。
- 与 tactile/WAM、主动视角、多智能体场景理解、Sim2Real、人形导航等主线交叉。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.20761](https://arxiv.org/abs/2609.20761) |
| **开源** | **待发布** |
| **策展摘要** | 轻量触觉 WAM：共享 latent + vision-tactile-to-action flow matching；9 仿真 + 5 真机；11.9 ms 推理、真机 SR 相对 +29.4%。 |


## 源码运行时序图

**不适用**（截至 2026-09-18 项目页/arXiv 未发布可运行官方代码）。

## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 SR/延迟/路径长度等 headline 数字。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**：11.9 ms 与相对 +29.4% 未与下列各页核对同一评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [LaWAM](./paper-lawam.md) | 同为「给 WAM 减负」，减的维度不同：LaWAM 把未来观测从像素换成潜空间 subgoal（省生成开销）；Agile-WAM 省的是模型规模，并**反向加了一路触觉**。同样低延迟，不是同一条路线 |
| [ParticleSplat](./paper-particlesplat.md) | 同批次里另一条「在动作头之前先补结构」：ParticleSplat 补对象中心 3D 几何，Agile-WAM 补接触时的触觉量。对应两类失败——看不清物体 vs 摸不出接触状态 |
| [PreDE](./paper-prede.md) | 同批次里另一条 WAM 部署成本路线，但**发力阶段不同**：Agile-WAM 在设计期做轻量化，PreDE 在部署期筛量化配置。两者正交，可叠加 |
| [World Action Models](../concepts/world-action-models.md) | 该页给 WAM 概念谱系；Agile-WAM 属「多模态 + flow matching 动作生成」一支，与纯视觉 WAM 的取舍是**传感器栈成本 vs 接触任务覆盖** |
| [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md) | 同批次横向对照入口：本文列 **深读** 档位，其余九篇分工见该页索引表 |

## 结论

**Agile-WAM 代表「深读」档位的 wam 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**待发布**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [Agile-WAM](./paper-agile-wam.md) / [INSPECT](./paper-inspect-view-selection.md) 等形成「触觉 WAM → 主动视角 → 系统平台」阅读链。
3. 若做工程选型，先对齐传感器栈与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- ./paper-lawam.md
- [manipulation](../tasks/manipulation.md)
- [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md)

## 参考来源

- [agile_wam_arxiv_2609_20761.md](../../sources/papers/agile_wam_arxiv_2609_20761.md)
- [wechat_embodied_station_10_papers_contact_wm_2026-09-18.md](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)
- [arXiv:2609.20761](https://arxiv.org/abs/2609.20761)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.20761)
