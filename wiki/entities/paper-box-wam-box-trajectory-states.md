---
type: entity
tags: [paper, world-action-models, object-centric, bounding-box, compact-state-representation, manipulation]
status: complete
updated: 2026-10-22
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

## 评测与结果

以下数字来自论文 HTML 版表格（arXiv:2610.17526）：

| 设定 | 指标 | 对照 | BoxWAM |
|------|------|------|--------|
| RoboTwin2.0 单任务，9 个任务平均（clean） | 成功率 % | Flow（同骨干、只预测动作）73.2；π0 71.4；DP3 81.2 | 78.3 |
| RoboTwin2.0 单任务，9 个任务平均（randomized） | 成功率 % | Flow 32.6；π0 33.1 | 37.8 |
| RoboTwin2.0 多任务微调（2,500 条示范，关节位置控制） | 成功率 % clean / randomized | π0.5：52.6 / 26.9 | π0.5 + BoxWAM：61.5 / 34.0 |
| 同上 | 推理延迟 ms | π0.5：55.3 | 57.7 |
| 真机三任务（煮咖啡、浇花、放置玩偶）平均 | 成功率与进度率均值 % | Diffusion Policy 23.7；Flow 51.1 | 60.9 |

- BoxWAM 与 Flow 每个任务、每种设定评测 3 个种子 × 100 次；其他单任务基线取自 RoboTwin2.0 排行榜，评测口径不完全相同。
- 在 π0.5 上加框轨迹预测只增加约 2.4 ms 延迟，这是摘要"显著降低计算开销"在多任务设定下的具体体现；与全帧视频预测的延迟对比见论文消融表。

## 与其他工作对比

| 路线 | 预测的未来 | 与 BoxWAM 的差异 |
|------|------------|------------------|
| 只预测动作的策略（论文 Flow 基线、[Diffusion Policy](../methods/diffusion-policy.md)） | 无 | 同骨干下 BoxWAM 在单任务 clean / randomized 两种设定都更高，说明未来状态监督本身有用 |
| 基于视频的 [世界动作模型](../concepts/world-action-models.md) | 稠密未来帧 | 监督信号更丰富但训练与推理更贵；BoxWAM 只保留任务相关物体的有向框轨迹 |
| [生成式世界模型](../methods/generative-world-models.md) | 可用于想象与评估的完整场景 | 目标是通用模拟；BoxWAM 只把未来预测当作动作生成的辅助条件 |
| [π0](./paper-pi0.md) / [π0.5](./paper-pi05-open-world-vla.md) | 无（预训练 VLA） | BoxWAM 可作为附加预测头接到 π0.5 上微调，而不是替代它 |

前两行的数值对比来自论文同一 RoboTwin2.0 协议；与视频世界模型和生成式世界模型的比较是表示层面的定性区分。

## 结论

**一句话总判：对操作任务而言，物体中心的框轨迹可能是比全帧视频更划算的未来预测目标，但仍需核对框的标注与估计成本。**

1. **紧凑表示可替代稠密预测** — 摘要以计算成本为主要动机。
2. **物体中心的信息足够多数任务** — 摘要认为框轨迹已覆盖关键物理变化（推测：适用范围仍依赖任务）。
3. **框的来源是隐含依赖** — 需要确认框是如何获得的（摘要未展开）。

## 局限与风险

- 框只描述物体刚体式运动，对柔性物体与复杂接触的表达能力未知。
- RoboTwin2.0 单任务中 BoxWAM 的 clean 平均（78.3）仍低于使用 RGBD 的 DP3（81.2），优势主要在 RGB 输入与随机化设定。

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
