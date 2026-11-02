---
type: entity
tags:
  - paper
  - quadruped
  - locomotion
status: complete
updated: 2026-11-02
arxiv: "2610.36852"
project_id: blackbox-policy-sensor-diagnosis-repair
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/blackbox-policy-sensor-diagnosis-repair_arxiv_2610_36852.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "黑箱策略传感器诊断修复：黑盒机器人策略失效时：基于透明代理的在线传感器诊断与修复（arXiv:2610.36852）：用显式分离周期步态与传感器修正项的透明代理近似策略，再利用代理梯度诊断并修正输入；在 Go2 上将故障条件成功数由 259/640 提高到 382/640。"
---

# 黑箱策略传感器诊断修复：黑盒机器人策略失效时：基于透明代理的在线传感器诊断与修复

**黑箱策略传感器诊断修复**（*When Black-Box Robot Policies Fail: Online Sensor Diagnosis and Repair with a Transparent Surrogate*，[arXiv:2610.36852](https://arxiv.org/abs/2610.36852)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**黑箱策略传感器诊断修复：黑盒机器人策略失效时：基于透明代理的在线传感器诊断与修复——用显式分离周期步态与传感器修正项的透明代理近似策略，再利用代理梯度诊断并修正输入；在 Go2 上将故障条件成功数由 259/640 提高到 382/640。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：冻结的黑盒运动策略遇到传感器故障时，无法通过访问策略内部结构定位和修复错误输入。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.36852](https://arxiv.org/abs/2610.36852) · [PDF](https://arxiv.org/pdf/2610.36852) |
| **作者（arXiv）** | Ziegler, Marc, Özen, Batu, Mentes, Sami, Waschulzik, Thomas, Knoll, Alois |
| **机构（周报所述，未逐条核实）** | 慕尼黑工业大学、Siemens Mobility |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 用显式分离周期步态与传感器修正项的透明代理近似策略，再利用代理梯度诊断并修正输入；在 Go2 上将故障条件成功数由 259/640 提高到 382/640。 |

### arXiv 摘要节选（英文原文）

> Robot foundation models may generalize across tasks and embodiments, but their actions still depend on sensors that can fail. When the policy is frozen or inaccessible, can an external monitor diagnose and repair corrupted inputs without access to its internals?

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**黑箱策略传感器诊断修复：黑盒机器人策略失效时：基于透明代理的在线传感器诊断与修复。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [locomotion](../tasks/locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [blackbox-policy-sensor-diagnosis-repair_arxiv_2610_36852.md](../../sources/papers/blackbox-policy-sensor-diagnosis-repair_arxiv_2610_36852.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.36852](https://arxiv.org/abs/2610.36852)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.36852)
