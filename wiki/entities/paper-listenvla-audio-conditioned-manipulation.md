---
type: entity
tags: [paper, vla, audio, multimodal, gr00t, sound-guided-manipulation]
status: complete
updated: 2026-10-22
project_id: listenvla
arxiv: "2610.16313"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
sources:
  - ../../sources/papers/listenvla_arxiv_2610_16313.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "ListenVLA（arXiv:2610.16313）：以 GR00T N1.7 为底座的音频条件 VLA，把语义音频嵌入与波形声学线索作为动作生成的条件，并用音频–动作绑定目标对齐声音差异与动作差异。"
---

# ListenVLA：音频条件的机器人操作

**ListenVLA 让 VLA 把声音当作条件输入：某些物体状态或交互结果只能通过声音判断时，策略可以据此调整动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| GR00T | Generalist Robot 00 Technology | NVIDIA 的人形机器人基础模型系列，本文底座为其 N1.7 版本 |
| FiLM | Feature-wise Linear Modulation | 按特征做仿射调制的条件注入方式（推测：摘要中的 feature modulation 指此类做法） |

## 为什么重要

摘要指出，许多操作任务依赖无法仅从视觉可靠推断的听觉线索，有时必须通过物理交互才能激发。仅用视觉、语言和本体感知的策略会遗漏这类信息。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.16313](https://arxiv.org/abs/2610.16313) |
| 作者 | Jiaqi Fan, Rui Xu, Jiakun Xia, Xuyu Fan, Shuailei Ma, Yuefeng Wang, Qing Wang, Jiarui Zhang（arXiv 作者列表） |
| 机构 | 未核实（摘要页未列） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **音频表示：** 语义音频嵌入与波形派生的声学特征结合，并覆盖因果的音频历史。
- **注入方式：** 音频以 audio tokens 与 feature modulation 两种方式条件化连续动作生成。
- **音频–动作绑定目标：** 用上下文匹配的示范，把听觉差异与示范中的动作差异对齐，监督相邻步的 flow 速度预测差。

## 工程实践

- 底座为 GR00T N1.7，复现需要对应底座与音频前端。
- 源码运行时序图：**不适用**（未核查到官方代码）。

## 评测与结果

据 arXiv 摘要（2026-10-22 核对）：

| 设置 | 内容 |
|------|------|
| 新基准 LIBERO-Audio | 基于 [LIBERO](./libero-benchmark.md) 的 8 个任务，覆盖按声音选目标、响应听觉事件、通过物理交互激发声学证据三类 |
| 真机 | 3 个真实操作任务 |
| 结果 | 摘要仅称"表现优异"，未给成功率数字 |
| 消融 | 支持音频–动作绑定监督与音频调制两项设计的贡献 |
| 音频替换测试 | 替换音频输入后动作预测随之改变，表明策略确实使用了声音 |

## 与其他工作对比

| 路线 | 输入模态 | 与 ListenVLA 的差异 |
|------|----------|---------------------|
| [GR00T N1.7](./isaac-gr00t.md) 原底座 | 视觉、语言、本体感知 | ListenVLA 在其上增加音频 token 与特征调制，并加入音频–动作绑定目标 |
| 一般 [VLA](../methods/vla.md) | 视觉、语言（部分含本体感知） | 无法利用仅由声音揭示的状态；LIBERO-Audio 正是针对这类任务构建 |
| [触觉感知](../concepts/tactile-sensing.md) 增强策略 | 接触 / 力觉信号 | 同为借助交互激发的非视觉线索；音频可远距感知事件，触觉需直接接触 |

以上为方法路线层面的定性对比，摘要未报告与其他音频 VLA 的同协议数值比较。

## 结论

**一句话总判：当任务线索来自声音时，把音频作为与视觉并列的条件输入，并用示范显式绑定听觉差异与动作差异，是这篇工作的主要思路。**

1. **声音是视觉之外的状态信号** — 论文以需要敲击或交互才能听到的线索为动机。
2. **绑定目标需要匹配的示范** — 上下文匹配的示范是学习听觉–动作对应关系的前提。
3. **历史音频比单帧更有用** — 方法使用因果音频历史（具体收益摘要未给出）。

## 局限与风险

- 摘要未给出 LIBERO-Audio 与真机的成功率数字，本页不引用。
- 声学环境噪声与任务多样性对泛化的影响未知。

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [操作任务](../tasks/manipulation.md)
- [触觉感知](../concepts/tactile-sensing.md)（同为借助非视觉交互信号的相关方向）

## 参考来源

- [论文归档](../../sources/papers/listenvla_arxiv_2610_16313.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.16313](https://arxiv.org/abs/2610.16313)
