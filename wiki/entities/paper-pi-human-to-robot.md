---
type: entity
tags: [paper, vla, human-to-robot, egocentric, manipulation, physical-intelligence, georgia-tech]
title: π 系人视频到机器人迁移
status: complete
updated: 2026-09-28
arxiv: "2512.22414"
related:
  - ./paper-pi05-open-world-vla.md
  - ../overview/ego-category-02-human-to-robot.md
  - ../methods/vla.md
  - ./pi-robot-olympics.md
sources:
  - ../../sources/papers/human_to_robot_arxiv_2512_22414.md
  - ../../sources/sites/pi-website-technical-articles.md
summary: "arXiv:2512.22414（RSS 2026）：π₀.₅ 预训练多样性够了之后，用人视频 3D 手部轨迹一起微调即可迁移，无需外观对齐。小规模预训练几乎没有这份收益。确认未开源。"
---

# 人视频到机器人：随预训练多样性出现的迁移

**Emergence of Human to Robot Transfer**（[arXiv:2512.22414](https://arxiv.org/abs/2512.22414)，RSS 2026，[项目页](https://www.pi.website/research/human_to_robot)）由 **物理智能（Physical Intelligence）** 与 **佐治亚理工学院（Georgia Tech）** 提出。问题不是再设计一种人手到夹爪的翻译，而是：π₀.₅ 这类通才在预训练变多样之后，能否把第一人称人视频当成又一种本体直接吃进去。

## 一句话定义

> **人机迁移被写成预训练多样性的函数：机器人数据够杂，微调时加入人视频才开始有用。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 被微调的 π₀.₅ |
| Ego | Egocentric video | 可穿戴相机的人视频，动作为 3D 手位置 |
| X-emb | Cross-embodiment | 预训练里除目标机器人以外的本体 |

## 为什么重要

人视频便宜，但外观和动作都和机器人不同。常见做法是遮挡身体、生成机器人手，或改硬件去贴近人。本文声称这些对齐可以先不做什么：把人视频按现有多本体格式加入微调，迁移会在预训练规模上来之后自己出现。这和「多采一点人视频就能替代机器人数据」不是同一句话。

## 核心原理

预训练只用机器人数据，并按场景–任务组合把多样性切成 0%、25%、50%、75%、100%，目标本体是 ARX 与移动 ARX；再加一档跨本体混合。人视频只在微调出现，动作是 3D 手部位置，图像不做特殊对齐。评测场景（例如按颜色分鸡蛋、整理特定梳妆台）只在人数据里出现，机器人数据只有相邻技能。

作者用骨干最后一层 token 的 t-SNE 说明：预训练不足时人与机器人特征分开；多样性提高后重叠增加。重叠来自更多机器人数据，不是因为预训练见过人视频。

## 评测

博客称四个泛化设置（bussing、香料、梳妆台、分鸡蛋）上，π₀.₅ 加入人数据后大约是不加的 2 倍。论文的尺度曲线更细：多样性 0% 和 25% 时人数据几乎无帮助；75%、100% 以及跨本体预训练后，增益变大。另有对照：Sort Eggs 与 Dresser 上人数据微调接近目标机器人自己的域内数据；Bussing 上机器人数据仍然更好。

## 结论

**人视频是预训练已经「见过足够多机器人」之后的增益项，不是跳过机器人数据的捷径。**

- 配方简单：共微调，3D 手位置，无生成式对齐
- 先看预训练多样性，再决定值不值得采人视频
- 有的任务人数据接近机器人域内数据，有的任务仍然差一截
- 特征重叠是解释，不是可直接监控的上线指标

## 源码运行时序图

**不适用**。截至 2026-09-28，项目页、arXiv 与 RSS 页都没有本文的代码、人视频或机器人数据。

## 局限与风险

- 确认未开源。「大约 2 倍」是作者四个设置的平均读法，单任务以论文曲线为准。
- 动作用的是手部 3D 位置，默认采集系统能提供这个量；只有 RGB 视频时本文配方不完整。
- 不要把它和 [Ego 分类 02](../overview/ego-category-02-human-to-robot.md) 里的其他对齐方法混成同一种算法。

## 关联页面

- [π₀.₅](./paper-pi05-open-world-vla.md)
- [Ego 分类：人→机器人](../overview/ego-category-02-human-to-robot.md)
- [VLA](../methods/vla.md)
- [Robot Olympics](./pi-robot-olympics.md)

## 参考来源

- [human_to_robot_arxiv_2512_22414](../../sources/papers/human_to_robot_arxiv_2512_22414.md)
- [PI 官网技术文章索引](../../sources/sites/pi-website-technical-articles.md)

## 推荐继续阅读

- [arXiv:2512.22414](https://arxiv.org/abs/2512.22414)
- [RSS 2026 页面](https://www.roboticsproceedings.org/rss22/p072.html)
