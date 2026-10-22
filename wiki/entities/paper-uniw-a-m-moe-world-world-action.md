---
type: entity
tags: [paper, world-model, world-action-models, mixture-of-experts, diffusion-transformer]
status: complete
updated: 2026-10-21
project_id: uniw-a-m
arxiv: "2610.18697"
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/uniw-a-m_arxiv_2610_18697.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "UniW(A)M（arXiv:2610.18697）：基于扩散 Transformer 的混合专家框架，用共享专家与可复用能力专家统一世界模型与多种世界动作模型（WAM）接口，部署时由插件式路由选择接口。"
---

# UniW(A)M：统一世界模型与世界动作模型的混合专家框架

**UniW(A)M 试图让预测环境变化的世界模型与输出机器人动作的 WAM 共享动力学知识，做法是用一组可跨接口复用的专家来承担不同能力。**

> **注意区分**：与 [UniWAM（arXiv:2610.02054）](./paper-uniwam-unified-world-action-model.md) 是两篇不同工作，名字相近但论文编号不同。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 世界模型，预测环境未来状态 |
| WAM | World Action Model | 世界动作模型，同时预测未来并生成动作 |
| MoE | Mixture of Experts | 混合专家，按输入路由到不同子网络 |

## 为什么重要

摘要指出，世界模型与世界动作模型都从具身轨迹学习物理动态与交互语义，但通常独立训练，限制了预测与控制之间的知识复用。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.18697](https://arxiv.org/abs/2610.18697) |
| 作者 | Jingbo Xu, Jianjie Fang, Zhuohang Li, Zihao Song, Zhaolu Wang, Xvyuan Liu, Xin Zhang, Haisheng Su（arXiv 作者列表） |
| 机构 | 未核实（摘要页未列） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **共享专家：** 捕获各接口共有的动力学。
- **能力专家：** 原子能力专家在不同接口间复用，每个接口保留自己的条件路径。
- **插件式接口路由：** 部署时按所需接口选择对应路径（摘要描述为 plug-in adaptive interface router）。

## 工程实践

- 未发现官方代码入口；源码运行时序图：**不适用**。

## 结论

**一句话总判：用共享专家加可复用能力专家统一世界模型与 WAM，是这篇工作的主要设计选择；它的价值取决于接口切换是否真的减少了训练成本。**

1. **动力学共享是出发点** — 两类模型学到的物理动态有重叠。
2. **能力专家跨接口复用** — 这是与各自独立训练的主要区别。
3. **需要看对照实验** — 摘要未给出成功率数字，不能仅凭结构判断收益。

## 局限与风险

- 与 UniWAM（2610.02054）的方法不同，比较时需按论文编号区分。
- 混合专家的路由稳定性与训练难度未在摘要中展开。

## 关联页面

- [世界动作模型](../concepts/world-action-models.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [UniWAM（另一篇同名相近工作）](./paper-uniwam-unified-world-action-model.md)

## 参考来源

- [论文归档](../../sources/papers/uniw-a-m_arxiv_2610_18697.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.18697](https://arxiv.org/abs/2610.18697)
