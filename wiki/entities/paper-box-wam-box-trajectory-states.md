---
type: entity
tags: [paper, world-action-models, object-centric, bounding-box, compact-state-representation, manipulation]
status: complete
updated: 2026-10-21
project_id: box-wam
arxiv: "2610.17526"
project: https://box-wam.github.io
code: https://github.com/quincy-u/box-wam
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/box-wam_arxiv_2610_17526.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "Box-WAM（arXiv:2610.17526）：用任务相关物体的有向边界框轨迹替代稠密未来帧，作为世界动作模型的紧凑预测目标；项目页链接的 GitHub 仓库可达（2026-10-21 核查）。"
---

# Box-WAM：边界框轨迹作为世界动作模型的紧凑状态

**Box-WAM 把"未来会怎样"压缩为任务相关物体的有向边界框轨迹，既保留物理变化信息，又避免全帧视频预测的计算开销。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界动作模型，同时预测未来状态与动作 |
| OBB | Oriented Bounding Box | 有向边界框，带旋转角的物体包围框 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

摘要指出，全帧视频预测在训练与推理时都昂贵。问题在于：什么样的紧凑状态表示能保留未来预测对规划与监督的收益。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.17526](https://arxiv.org/abs/2610.17526) |
| 作者 | Qinxi Yu, Wenzhou Ding, Gloria X. Wang, Albert J. Zhai, Shenlong Wang（arXiv 作者列表） |
| 机构 | 未核实（公众号标注"待核实"） |
| 开源状态 | **已链接仓库，可达**（2026-10-21 核查：项目页 box-wam.github.io 链接 github.com/quincy-u/box-wam，`git ls-remote` 可达；未阅读仓库内容，是否含训练与推理入口未核实） |
| 项目页 | <https://box-wam.github.io> |

## 核心原理

- **预测目标：** 用任务相关物体的边界框轨迹替代稠密未来帧。
- **动机：** 框轨迹保留机器人动作引起的关键物理变化（物体位置与运动）。
- **效果（摘要）：** 框轨迹预测显著降低计算开销，同时提升操作表现。

## 工程实践

- 仓库可达，但入口与依赖未核对。
- 源码运行时序图：**不适用**（未读取仓库入口代码，无法据实绘制运行时序）。

## 结论

**一句话总判：对操作任务而言，物体中心的框轨迹可能是比全帧视频更划算的未来预测目标，但仍需核对框的标注与估计成本。**

1. **紧凑表示可替代稠密预测** — 摘要以计算成本为主要动机。
2. **物体中心的信息足够多数任务** — 摘要认为框轨迹已覆盖关键物理变化（推测：适用范围仍依赖任务）。
3. **框的来源是隐含依赖** — 需要确认框是如何获得的（摘要未展开）。

## 局限与风险

- 框只描述物体刚体式运动，对柔性物体与复杂接触的表达能力未知。
- 摘要未给出具体成功率数字，本页不引用。

## 关联页面

- [世界动作模型](../concepts/world-action-models.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [操作任务](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/box-wam_arxiv_2610_17526.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.17526](https://arxiv.org/abs/2610.17526)
- [项目页](https://box-wam.github.io)
