---
type: entity
tags: [paper, vla, bimanual-manipulation, dexterous-hands, action-representation, action-tokenization]
status: complete
updated: 2026-10-21
project_id: coordi-pi
arxiv: "2610.19753"
related:
  - ../methods/vla.md
  - ../concepts/dexterous-kinematics.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/coordi-pi_arxiv_2610_19753.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "Coordi-π（arXiv:2610.19753）：面向双臂灵巧操作的协调感知动作表示，把左右手动作拆成独立 token 并交错排列，用可学习的手部身份嵌入与同步信息显式表达双手协作。"
---

# Coordi-π：双臂灵巧操作的协调感知动作表示

**Coordi-π 不再把左右手动作拼成一个向量，而是把每只手的动作作为独立 token 交错排列，让双手关系出现在动作序列里。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| DoF | Degree of Freedom | 自由度 |
| ID | Identity | 此处指手部身份嵌入 |

## 为什么重要

摘要指出，现有双臂 VLA 通常在每个控制步把左右臂–手动作拼接成一个动作向量，跨手交互因此只能隐含地存在于单个向量中。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.19753](https://arxiv.org/abs/2610.19753) |
| 作者 | Yiming Meng, Yanming Shao, Weijie Shen, Kunyang Lin, Jiayu Sun, Ping Luo, Yao Mu（arXiv 作者列表） |
| 机构 | 未核实（公众号标注"待核实"） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **交错动作序列：** 左右手动作各自成为 token，按序排列。
- **手部身份嵌入：** 可学习嵌入区分两只手。
- **时序同步：** 摘要提到序列–控制的同步编码，用于对齐两手的时间关系。

## 工程实践

- 未发现官方代码入口；源码运行时序图：**不适用**。

## 结论

**一句话总判：双手协作需要在动作表示层显式建模，而不是只靠网络从拼接向量中隐式学习。**

1. **表示形式影响跨手建模** — 交错 token 把双手关系暴露在序列中。
2. **身份与同步信息需要显式编码** — 摘要的两个嵌入即为此设计。
3. **收益需看对照** — 摘要未给出成功率数字。

## 局限与风险

- 序列长度随手部自由度增加，计算成本未在摘要中给出（推测：会增长）。

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [灵巧运动学](../concepts/dexterous-kinematics.md)
- [操作任务](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/coordi-pi_arxiv_2610_19753.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.19753](https://arxiv.org/abs/2610.19753)
