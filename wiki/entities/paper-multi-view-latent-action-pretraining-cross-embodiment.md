---
type: entity
tags:
  - paper
  - cross-embodiment
  - manipulation
status: complete
updated: 2026-10-26
arxiv: "2610.27794"
project_id: multi-view-latent-action-pretraining-cross-embodiment
project: https://learnsyslab.github.io/multiview_latent_action_pretraining/
related:
  - ../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/multi-view-latent-action-pretraining-cross-embodiment_arxiv_2610_27794.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md
summary: "Multi-View LAPA：面向跨本体机器人视频学习的多视角潜在动作预训练（arXiv:2610.27794）：研究通过多视角视频学习潜在动作表示，使模型能够利用不同视角的运动信息进行预训练，并将学到的动作表示用于跨机器人本体学习。"
---

# Multi-View LAPA：面向跨本体机器人视频学习的多视角潜在动作预训练

**Multi-View LAPA**（*Multi-View Latent Action Pretraining for Cross-Embodiment Robot Learning from Videos*，[arXiv:2610.27794](https://arxiv.org/abs/2610.27794)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**Multi-View LAPA：面向跨本体机器人视频学习的多视角潜在动作预训练——研究通过多视角视频学习潜在动作表示，使模型能够利用不同视角的运动信息进行预训练，并将学到的动作表示用于跨机器人本体学习。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：不同机器人动作空间不一致，而视频通常不包含可直接使用的机器人控制标签。单一视角还可能无法完整观察操作运动。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.27794](https://arxiv.org/abs/2610.27794) · [PDF](https://arxiv.org/pdf/2610.27794) |
| **作者（arXiv）** | Oliver Hausdörfer, Felix Minzenmay, Mani Majd, Maximilian P. Christof, Shiyuan Zhang, Angela P. Schoellig |
| **机构（周报所述，未逐条核实）** | 待核实（作者：Oliver Hausdörfer、Felix Minzenmay、Angela P. Schoellig 等）。 |
| **arXiv 评论栏** | Website: this https URL |
| **开源** | arXiv 摘要页/评论给出项目页或外链（https://learnsyslab.github.io/multiview_latent_action_pretraining/）；未发现代码仓库链接（2026-10-26） |
| **周报所述创新点** | 研究通过多视角视频学习潜在动作表示，使模型能够利用不同视角的运动信息进行预训练，并将学到的动作表示用于跨机器人本体学习。 |

### arXiv 摘要节选（英文原文）

> Action-free videos are an appealing substitute for costly teleoperated data. Latent action models (LAMs) turn videos into a supervision signal by inferring abstract actions between frames.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出项目页或外链（https://learnsyslab.github.io/multiview_latent_action_pretraining/）；未发现代码仓库链接，无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出项目页或外链（https://learnsyslab.github.io/multiview_latent_action_pretraining/）；未发现代码仓库链接 |

## 结论

**Multi-View LAPA：面向跨本体机器人视频学习的多视角潜在动作预训练。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-manipulation-technology-map](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [multi-view-latent-action-pretraining-cross-embodiment_arxiv_2610_27794.md](../../sources/papers/multi-view-latent-action-pretraining-cross-embodiment_arxiv_2610_27794.md)
- [wechat_senlanke_weekly_manipulation_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
- [arXiv:2610.27794](https://arxiv.org/abs/2610.27794)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.27794)
