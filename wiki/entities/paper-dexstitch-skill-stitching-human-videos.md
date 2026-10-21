---
type: entity
tags: [paper, dexterous-manipulation, human-video, skill-segmentation, skill-composition, inpainting]
status: complete
updated: 2026-10-21
project_id: dexstitch
arxiv: "2610.17789"
related:
  - ../methods/imitation-learning.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/dexterous-kinematics.md
  - ../tasks/manipulation.md
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

## 结论

**一句话总判：从人类视频中同时学习技能与过渡，并用接触信息衔接，是组合灵巧技能的一条数据路线。**

1. **切分是前提** — 技能与过渡都需要标签才能组合。
2. **接触信息要随数据保留** — 摘要强调逐指接触记录。
3. **拼接效果需看长时程成功率** — 摘要未给出数字。

## 局限与风险

- 技能分类器的错误会传播到后续拼接（推测）。
- 补全质量依赖机器人渲染与视频分割的准确性（推测）。

## 关联页面

- [模仿学习](../methods/imitation-learning.md)
- [接触丰富操作](../concepts/contact-rich-manipulation.md)
- [灵巧运动学](../concepts/dexterous-kinematics.md)

## 参考来源

- [论文归档](../../sources/papers/dexstitch_arxiv_2610_17789.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.17789](https://arxiv.org/abs/2610.17789)
