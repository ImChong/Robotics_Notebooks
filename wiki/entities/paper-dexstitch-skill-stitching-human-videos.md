---
type: entity
tags: [paper, dexterous-manipulation, human-video, skill-segmentation, skill-composition, inpainting]
status: complete
updated: 2026-10-22
project_id: dexstitch
arxiv: "2610.17789"
related:
  - ../methods/imitation-learning.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/dexterous-kinematics.md
  - ../tasks/manipulation.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/dexstitch_arxiv_2610_17789.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "DexStitch（arXiv:2610.17789）：把长时程人类视频切分为技能与过渡，经接触感知的机器人补全转为对齐的观测–动作–技能数据，并把学到的技能按指定顺序拼接。"
---

# DexStitch：从人类视频学习并组合灵巧技能

**DexStitch 把长视频切成技能段与过渡段，再把人类手部示范转换成机器人可用的数据，使单个技能可以按需串联。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| — | Robotic Inpainting | 机器人补全：在视频中把人手替换为机器人并补全被遮挡的动作（据摘要） |
| — | Contact-aware | 接触感知，逐指记录人与物体的接触关系 |

## 为什么重要

摘要指出，技能之间的衔接需要满足接触关系与运动连续性。仅有单个技能的数据不足以覆盖长时程任务。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.17789](https://arxiv.org/abs/2610.17789) |
| 作者 | Byeonggyeol Choi, Seongmin Jung, Unghui Nam, Youngbo Song, Sunhong Min, MinHyung Lee, Byonghyo Shim, Yongseok Lee（arXiv 作者列表） |
| 机构 | 未核实（公众号标注"待核实"） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

1. **技能分割：** 技能分类器把视频切分为操作技能与过渡段。
2. **转换：** 把人类示范转为机器人观测与动作，同时保留技能标签；转换时逐指记录人–物接触。
3. **拼接：** 把学到的技能按请求的顺序串联。

## 工程实践

- 未发现官方代码入口；源码运行时序图：**不适用**。

## 评测与结果

据 arXiv 摘要与 HTML 正文（Table I–III）：

| 评测 | 设置 | 结果 |
|------|------|------|
| 技能学习与组合（Table I） | 四个家务技能；单技能每项 5 次，四技能序列 10 次 | 单技能平均 80.0%，序列总成功率 77.5% |
| 技能分类器（Table II） | 留出会话 24 段人类视频、18,792 帧 | 帧准确率 92.4%、Macro-F1 92.7、Edit 99.6、F1@50 99.0；恢复 242 段（真值 241） |
| 消融（Table III，四技能序列） | 均为 90% 遥操作微调比例 | 无人类数据 35.0%、无遥操作数据 5.0%、去掉 transition 技能 10.0%、去掉 next-skill 条件 7.5%；遥操作比例 50% / 75% / 100% 分别为 10.0% / 20.0% / 50.0% |

微调时保留约 10% 合成数据（90% 遥操作）效果最好，纯遥操作反而降到 50.0%。

## 与其他工作对比

| 对照对象 | 论文设置 | 与 DexStitch 的差异 |
|----------|----------|---------------------|
| UniSkill（技能学习与组合基线） | 同样的预训练 / 微调步数与数据，仅技能表示不同 | 单技能 45.0%，但没有显式过渡机制，四技能序列 0.0%；DexStitch 为 80.0% / 77.5% |
| [Masquerade](./paper-notebook-masquerade-learning-from-in-the-wild-human-video.md) | 人→机器人视频转换，**定性**对比 | 同为把人类视频补全为机器人画面；DexStitch 额外用逐指接触引导遮挡顺序与手部重定向 |
| Qwen2.5-VL-7B / InternVL3-8B（零样本） | 同一闭集技能问题、同帧同协议 | 帧准确率 65.8% / 57.8%、F1@50 41.0 / 28.5；论文认为差距主要在过渡段（F1 0.41 / 0.18 vs 0.89） |

DexStitch 的分类器基于 [V-JEPA 2](./paper-vjepa2.md) 特征加 MLP。以上均为论文同协议结果，只覆盖四个家务技能，不能外推到更长或更多样的任务。

## 结论

**一句话总判：从人类视频中同时学习技能与过渡，并用接触信息衔接，是组合灵巧技能的一条数据路线。**

1. **切分是前提** — 技能与过渡都需要标签才能组合。
2. **接触信息要随数据保留** — 摘要强调逐指接触记录。
3. **过渡需要显式建模** — 序列成功率 77.5%，而无显式过渡的 UniSkill 在四技能序列上为 0%。

## 局限与风险

- 技能分类器的错误会传播到后续拼接（推测）。
- 补全质量依赖机器人渲染与视频分割的准确性（推测）。

## 关联页面

- [模仿学习](../methods/imitation-learning.md)
- [接触丰富操作](../concepts/contact-rich-manipulation.md)
- [灵巧运动学](../concepts/dexterous-kinematics.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 注意区分：本文的 skill segmentation 是时间轴上的技能切分；把人类视频转为机器人观测时依赖的手/物体视觉分割与补全，才落在感知栈的 ②/③ 层

## 参考来源

- [论文归档](../../sources/papers/dexstitch_arxiv_2610_17789.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.17789](https://arxiv.org/abs/2610.17789)
