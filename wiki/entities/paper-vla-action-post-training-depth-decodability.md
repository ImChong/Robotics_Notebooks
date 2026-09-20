---
type: entity
tags:
  - paper
  - vla
  - analysis
  - depth
  - representation
status: complete
updated: 2026-09-19
arxiv: "2608.08904"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/vla_action_post_training_depth_decodability_arxiv_2608_08904.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "诊断 VLM→VLA 动作后训练如何削弱深度可解码性：逐层发现全层退化且后层 MLP 写入深度信息受干扰；移除相关写入可恢复大部分末端损失（EMR@ECCV 2026）。"
---

# VLA Depth Decodability（arXiv:2608.08904）

**VLA Depth Decodability**（[arXiv:2608.08904](https://arxiv.org/abs/2608.08904)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **分析诊断、空间感知** 段。

## 一句话定义

**诊断 VLM→VLA 动作后训练如何削弱深度可解码性：逐层发现全层退化且后层 MLP 写入深度信息受干扰；移除相关写入可恢复大部分末端损失（EMR@ECCV 2026）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 诊断 VLM→VLA 动作后训练如何削弱深度可解码性：逐层发现全层退化且后层 MLP 写入深度信息受干扰；移除相关写入可恢复大部分末端损失（EMR@ECCV 2026）。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.08904](https://arxiv.org/abs/2608.08904) |
| **开源** | **待核实** |
| **文内评测** | LIBERO |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** LIBERO
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [Depth-Wise Probing Driving VLA](./paper-depth-wise-probing-driving-vla.md) | 同批两篇**逐层探针**诊断，结论指向相反的用法：本文探深度信息，发现动作后训练令其**全层退化**、后层 MLP 的写入受干扰，指向把被削掉的表征**修回来**；那篇探规划 token，发现浅层已够，指向把多余的层**剪掉**。同一诊断工具，一个做修复，一个做裁剪 |
| [AnyCamVLA](./paper-anycam-vla.md) / [WNM-3D](./paper-wnm-3d-vln.md) | 同批「空间感知」段两条是在**补几何信息**（测试期视角归一化 / 3D 场景条件），本文解释的是**为什么会缺**——不是没喂进去，而是动作后训练把它写没了。三页合读才给得出「该补还是该护」的判据 |
| [VLA](../methods/vla.md) | 该页讲 VLM→VLA 的后训练路径；本文是这条路径的一份**代价清单**：动作能力上来的同时，预训练里的深度表征被牺牲掉一部分 |
| [Sim2Real](../concepts/sim2real.md) | 深度表征退化会在换场景、换相机时先暴露；读本文应把它当**泛化退化的一个可测机制**，而不是单纯的可解释性结论 |

## 结论

**VLA Depth Decodability 适合作为本期「分析诊断、空间感知」路线的快速索引页。**

1. 核心贡献：诊断 VLM→VLA 动作后训练如何削弱深度可解码性：逐层发现全层退化且后层 MLP 写入深度信息受干扰；移除相关写入可恢复大部分末端损失（EMR@ECCV 2026）。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2608.08904](https://arxiv.org/abs/2608.08904)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.08904)
