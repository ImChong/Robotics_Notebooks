---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-10-26
arxiv: "2610.26170"
project_id: feature-level-lfd-prompt-detection-nao
related:
  - ../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/feature-level-lfd-prompt-detection-nao_arxiv_2610_26170.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
summary: "特征级示范学习：基于特征级感知与提示式目标检测的人形机器人数据高效示范学习（arXiv:2610.26170）：从视频中提取人体关节、提示词检测结果和三维物体位置等紧凑特征，再预测 NAO 的关节轨迹；在极少示范条件下完成推物和抓放任务。"
---

# 特征级示范学习：基于特征级感知与提示式目标检测的人形机器人数据高效示范学习

**特征级示范学习**（*Data-Efficient Learning from Demonstration for Humanoid Robots via Feature-Level Perception and Prompt-Based Object Detection*，[arXiv:2610.26170](https://arxiv.org/abs/2610.26170)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**特征级示范学习：基于特征级感知与提示式目标检测的人形机器人数据高效示范学习——从视频中提取人体关节、提示词检测结果和三维物体位置等紧凑特征，再预测 NAO 的关节轨迹；在极少示范条件下完成推物和抓放任务。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：从原始图像端到端学习人形操作通常需要大量示范和计算资源。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.26170](https://arxiv.org/abs/2610.26170) · [PDF](https://arxiv.org/pdf/2610.26170) |
| **作者（arXiv）** | Keyhan Rayati, AliReza Beigy, Anoosheh Saadati, Pourya Shahverdi, Mehdi Tale Masouleh, Ahmad Kalhor, Wing-Yue Geoffrey Louie |
| **机构（周报所述，未逐条核实）** | 德黑兰大学、奥克兰大学 |
| **arXiv 评论栏** | 6 pages, 4 figures, 2 tables. Presented at IEEE ISAIA 2026 |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26） |
| **周报所述创新点** | 从视频中提取人体关节、提示词检测结果和三维物体位置等紧凑特征，再预测 NAO 的关节轨迹；在极少示范条件下完成推物和抓放任务。 |

### arXiv 摘要节选（英文原文）

> Learning from Demonstration (LfD) allows robots to learn new skills from human examples; however, many vision-based methods require large datasets and resource-intensive end-to-end training. This paper introduces a data-efficient LfD framework for humanoid robots that learns manipulation skills from compact, task-specific features instead of raw images.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**特征级示范学习：基于特征级感知与提示式目标检测的人形机器人数据高效示范学习。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [feature-level-lfd-prompt-detection-nao_arxiv_2610_26170.md](../../sources/papers/feature-level-lfd-prompt-detection-nao_arxiv_2610_26170.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
- [arXiv:2610.26170](https://arxiv.org/abs/2610.26170)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.26170)
