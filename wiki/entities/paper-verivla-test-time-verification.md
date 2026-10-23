---
type: entity
tags: [paper, vla, flow-matching, test-time-scaling, world-model, verification, jepa]
status: complete
updated: 2026-10-22
project_id: verivla
arxiv: "2610.21985"
related:
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ../concepts/flow-matching-embodied-policy.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/verivla_arxiv_2610_21985.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "VeriVLA（arXiv:2610.21985）：免训练的测试时验证，用两个初始采样检测关键状态，再由 JEPA 式潜在世界模型按预测任务进度对候选动作块排序。"
---

# VeriVLA：Flow-VLA 的自适应测试时验证

**VeriVLA 利用 Flow-VLA 生成的随机性：只在判断为关键的步骤上，从多个候选动作块中挑出预测进度最好的一个。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| JEPA | Joint-Embedding Predictive Architecture | 在潜在空间预测未来表征、不重建像素的架构 |
| WM | World Model | 世界模型，用于预测动作后果 |

## 为什么重要

精细操作中，一次次优动作可能不可逆地破坏任务进度。摘要指出，Flow-VLA 对同一观测的不同噪声种子会给出质量不同的动作块，但测试时通常直接提交一个样本，没有质量估计。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.21985](https://arxiv.org/abs/2610.21985) |
| 作者 | Xueli Sun, Yuxiang Lu, Yantong Liu, Xianzhe Fan, Xi Chen, Zhe Liu, Hengshuang Zhao（arXiv 作者列表） |
| 机构 | 未核实（摘要页未列） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **两轴触发：** 用两个初始样本衡量块内时间变化（within-chunk temporal variation）与跨样本方差，检测关键状态；非关键状态不额外计算。
- **潜在世界模型排序：** 触发后，JEPA 式世界模型在潜在空间预测每个候选动作块的未来状态与任务进度变化，并据此排序。
- **不重建像素：** 验证在潜在空间完成，避免像素重建的开销。

## 工程实践

- 免训练，可挂在已有 Flow-VLA 的采样流程之后（推测：需要模型支持多次采样）。
- 源码运行时序图：**不适用**（未核查到官方代码）。

## 评测与结果

以下数字来自 arXiv 摘要与论文 HTML 版 Table 1 等，本库未复现。

| 评测 | 底座策略 | 论文报告 |
|------|----------|----------|
| 真机三任务（插笔、移液、放苹果；AgileX PiperX，1 前置 + 2 腕部相机） | π0.5 | 平均完成分 50.8% → 65.5%（插笔 45.0→60.0、移液 52.5→63.5、放苹果 55.0→73.0） |
| 延迟 | 单张 RTX 4090 | 每次触发的采样 + 验证增加 326 ms（不含触发器本身） |
| [RoboTwin 2.0](./robotwin.md) Clean（5 个低基线精细任务） | Hy-Embodied-0.5-VLA | 平均成功率 66.4% → 73.6% |
| RoboTwin 2.0 Clean→Randomized | 同上 | 62.8% → 69.6% |
| [LIBERO](./libero-benchmark.md) | π0.5 等两种底座、40 任务 | 96.9% → 97.6% |
| 消融 | 真机三任务 | 去掉世界模型平均 55.2%，去掉预训练 59.5%，完整方法 65.5% |

## 与其他工作对比

| 方法（论文基线） | 验证器类型 | 真机平均完成分 |
|------------------|------------|----------------|
| π0.5 单次采样 | 无 | 50.8% |
| + RoboMonkey | VLM 打分验证器（训练目标改为进度增量） | 56.0% |
| + GPC-RANK | 像素级世界模型验证器 | 52.8% |
| + VeriVLA | JEPA 式潜在世界模型 + 两轴触发 | 65.5% |

上表为论文同一真机协议下的结果。作者指出 RoboMonkey 在仅需粗定位的任务（放苹果）上有竞争力，但在精细操作上落后。思路相近的库内页面：[WAM 测试时引导](./paper-wam-ttt-human-video-test-time-steering.md)、[JEPA-WAM](./paper-jepa-wam.md)，二者未与 VeriVLA 共享评测口径。

## 结论

**一句话总判：对 Flow-VLA 而言，测试时多采样加世界模型排序是一条不改训练的可选增益路径，但增益应在关键步骤上计算才划算。**

1. **随机性本身是信号** — 同一观测的多次采样质量差异，可用于触发验证。
2. **验证只在关键步触发** — 两轴触发器的作用是控制额外计算量。
3. **潜在空间比像素空间更便宜** — 摘要以此作为效率论点。

## 局限与风险

- 所有数字为论文自报；真机仅三任务，每次触发额外 326 ms 对高频控制是否可接受需按任务评估。
- 世界模型的预测误差会直接影响排序质量。

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [世界动作模型](../concepts/world-action-models.md)
- [Flow matching 具身策略](../concepts/flow-matching-embodied-policy.md)

## 参考来源

- [论文归档](../../sources/papers/verivla_arxiv_2610_21985.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.21985](https://arxiv.org/abs/2610.21985)
