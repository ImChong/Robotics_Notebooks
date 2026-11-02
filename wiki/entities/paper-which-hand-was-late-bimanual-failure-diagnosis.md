---
type: entity
tags:
  - paper
  - bimanual
  - manipulation
status: complete
updated: 2026-11-02
arxiv: "2610.37533"
project_id: which-hand-was-late-bimanual-failure-diagnosis
related:
  - ../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/which-hand-was-late-bimanual-failure-diagnosis_arxiv_2610_37533.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
summary: "Which Hand Was Late：哪只手延迟了？双臂工具使用中的失败诊断与恢复（arXiv:2610.37533）：系统分析不同传感器配置能够识别哪些失败原因，并建立基于诊断结果的恢复策略。实验发现，部分失败原因在被动观测下无法唯一确定，因此恢复策略需要根据可观测性决定直接修正还是进一步获取信息。"
---

# Which Hand Was Late：哪只手延迟了？双臂工具使用中的失败诊断与恢复

**Which Hand Was Late**（*Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use*，[arXiv:2610.37533](https://arxiv.org/abs/2610.37533)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**Which Hand Was Late：哪只手延迟了？双臂工具使用中的失败诊断与恢复——系统分析不同传感器配置能够识别哪些失败原因，并建立基于诊断结果的恢复策略。实验发现，部分失败原因在被动观测下无法唯一确定，因此恢复策略需要根据可观测性决定直接修正还是进一步获取信息。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：双臂工具操作的失败可能来自左右臂不同步、工具滑移或接触力不足。单一相机或单臂传感器未必能够区分这些原因。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.37533](https://arxiv.org/abs/2610.37533) · [PDF](https://arxiv.org/pdf/2610.37533) |
| **作者（arXiv）** | Bonin, Baptiste, Strickland, Caro, Durand, Audrey |
| **机构（周报所述，未逐条核实）** | 待核实（作者：Baptiste Bonin、Caro Strickland、Audrey Durand）。 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 系统分析不同传感器配置能够识别哪些失败原因，并建立基于诊断结果的恢复策略。实验发现，部分失败原因在被动观测下无法唯一确定，因此恢复策略需要根据可观测性决定直接修正还是进一步获取信息。 |

### arXiv 摘要节选（英文原文）

> Failure diagnosers for robot manipulation are increasingly vision--language models (VLMs), trained and evaluated on single-arm failures produced by perturbing a successful demonstration. Bimanual tool use adds failures of other kinds: a tool that slips out of its socket while still held, two correct gestures made at different times, and attempts that fail for two reasons at once.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**Which Hand Was Late：哪只手延迟了？双臂工具使用中的失败诊断与恢复。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-manipulation-technology-map](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [which-hand-was-late-bimanual-failure-diagnosis_arxiv_2610_37533.md](../../sources/papers/which-hand-was-late-bimanual-failure-diagnosis_arxiv_2610_37533.md)
- [wechat_senlanke_weekly_manipulation_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
- [arXiv:2610.37533](https://arxiv.org/abs/2610.37533)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.37533)
