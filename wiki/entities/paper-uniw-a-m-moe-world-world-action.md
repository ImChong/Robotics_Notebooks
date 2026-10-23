---
type: entity
tags: [paper, world-model, world-action-models, mixture-of-experts, diffusion-transformer]
status: complete
updated: 2026-10-22
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

## 评测与结果

以下数字来自 arXiv 摘要与论文 HTML 版正文，本库未复现。

| 评测 | 设置 | 论文报告 |
|------|------|----------|
| 真机三任务（叠衣、折盒、桌面清理） | 每任务 20 次 | 平均成功率 88.3%（IDM-WAM 接口），最强基线 WorldScape Policy 2.0 为 76.7%；四个 WAM 接口均高于该基线 |
| [RoboTwin 2.0](./robotwin.md) | 仅 clean 数据训练，50 任务平均 | Clean→Clean 84.38%（对比方法中排第二）；Clean→Randomized 66.60%，比所列最强基线 π0.5 高 20.60 个百分点 |
| [LIBERO](./libero-benchmark.md) | 标准套件 | 摘要列为评测基准之一（具体数值见论文表格，本页未摘录） |
| WorldArena（动作条件世界建模） | 三任务 EWMScore-P | 三个任务均为对比方法最高（对照含 Ctrl-World、Cosmos-Predict2.5） |
| PAIR 路由（分布外） | 冻结骨干，只训练接口路由 | 平均成功率 63.3% → 76.7% |
| 结构消融 | 同步数、同数据量的单一稠密 DiT（Wan2.1-I2V-14B 初始化） | 稠密基线最佳成功率仅 16.7%，联合训练 UniW(A)M 达 88.3% |

## 与其他工作对比

| 对照 | 与 UniW(A)M 的差异 |
|------|--------------------|
| [WorldScape Policy 2.0](./paper-worldscape-policy-2.md)（真机最强基线） | 单一策略接口；UniW(A)M 让四种 WAM 接口共享专家并按 chunk 切换 |
| [π0.5](./paper-pi05-open-world-vla.md)（RoboTwin C2R 基线） | 不显式预测未来；UniW(A)M 用动作条件世界模型协同训练所有动作接口 |
| [Ctrl-World](./paper-ctrl-world.md) / Cosmos-Predict2.5（WorldArena 基线） | 只做世界建模，不输出动作；UniW(A)M 在同一骨干内兼顾预测与控制 |
| [UniWAM（2610.02054）](./paper-uniwam-unified-world-action-model.md) | 名称相近的另一篇统一 WAM 工作，方法与评测口径不同，数值不可直接对比 |

前三行来自论文同一评测协议；与 UniWAM 的对比仅为定性区分。

## 结论

**一句话总判：用共享专家加可复用能力专家统一世界模型与 WAM，是这篇工作的主要设计选择；它的价值取决于接口切换是否真的减少了训练成本。**

1. **动力学共享是出发点** — 两类模型学到的物理动态有重叠。
2. **能力专家跨接口复用** — 这是与各自独立训练的主要区别。
3. **需要看对照实验** — 真机 88.3% 对最强基线 76.7%，同预算稠密 DiT 基线仅 16.7%，收益主要来自联合训练与专家共享（论文自报）。

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
