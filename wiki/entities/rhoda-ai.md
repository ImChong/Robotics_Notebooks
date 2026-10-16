---
type: entity
tags: [company, world-action-model, video-policy, foundation-policy, manipulation, tactile, rhoda-ai]
title: Rhoda AI（机器人基础模型公司）
status: complete
updated: 2026-10-16
summary: "Rhoda AI 以互联网规模视频预训练 + 闭环视频预测控制（Direct Video-Action，DVA）做机器人基础模型 FutureVision；2026-03 结束隐身并宣布 4.5 亿美元 A 轮，官网 3 篇研究博文覆盖 DVA、网络视频预训练 scaling 与触觉传感，全部闭源、数字为公司自报。"
related:
  - ./rhoda-direct-video-action.md
  - ./rhoda-scaling-web-video-pretraining.md
  - ./rhoda-dva-tactile-sensing.md
  - ../concepts/world-action-models.md
  - ../concepts/inverse-dynamics-model.md
  - ../concepts/embodied-scaling-laws.md
  - ../comparisons/robot-foundation-model-company-paths-2026.md
  - ./generalist-ai-robotics.md
  - ./dyna-2.md
sources:
  - ../../sources/sites/rhoda-ai.md
---

# Rhoda AI（机器人基础模型公司）

## 一句话定义

**Rhoda AI**：位于美国加州 Palo Alto 的商业公司，把机器人策略重述为**视频生成**——先在数亿条互联网视频上预训练因果视频模型，再用少量机器人数据后训练，并用逆动力学模型把预测画面转成动作，闭环运行（**DVA**，Direct Video-Action）；对外产品叙事是基础模型 **FutureVision**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DVA | Direct Video-Action | Rhoda 的专有架构：视频模型直接决定策略行为，逆动力学模型把预测画面转成动作 |
| IDM | Inverse Dynamics Model | 逆动力学模型；由「当前画面 + 预测未来画面」反推动作 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略；新闻稿以其为对照，称其难应付真实环境的变化 |
| DINO FD | DINO Fréchet Distance | Rhoda 用冻结 DINOv2 特征计算的预训练质量指标，见 [scaling 研究](./rhoda-scaling-web-video-pretraining.md) |
| PoC | Proof of Concept | 概念验证；DVA 博文把两个案例称为客户 PoC |

## 为什么重要

- **「视频即策略」的产业样本：** 与 [Generalist AI](./generalist-ai-robotics.md) 以真实交互数据预训练的路线不同，Rhoda 把主要预训练预算放在**网络视频**上、只在后训练阶段引入机器人动作；与 [Dyna-2](./dyna-2.md) 等同属闭源的产业路线，读法见 [世界动作模型](../concepts/world-action-models.md)。
- **公开了少见的真机 scaling 数据：** [网络视频预训练 scaling](./rhoda-scaling-web-video-pretraining.md) 在同一任务上只改模型规模与预训练算力，给出 200 多小时真机评测下的成功率曲线，可与 [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md) 对照。
- **从视觉扩到触觉：** [触觉 + DVA](./rhoda-dva-tactile-sensing.md) 把触觉当作与视频同构的「触觉视频」输入输出，延续同一套预测—逆动力学管线。

## 公司概况

| 项 | 内容 | 来源 |
|----|------|------|
| 公开亮相 | **2026-03-10**，新闻稿称「after 18 months in stealth」 | 官网新闻稿 |
| 成立时间 | **官方未写**；按 18 个月隐身期倒推约 2024 年下半年（**推测**）。路线数据以 2024 年作排序锚点 | 推测，见[来源归档](../../sources/sites/rhoda-ai.md) |
| 地点 | Palo Alto, CA（新闻稿发稿地，未明写为总部） | 官网新闻稿 |
| 领导层 | CEO / 联创 Jagdeep Singh；Chief Scientist / 联创 Eric Chan（新闻稿称 CSO，前 WorldLabs 生成模型架构师）；Gordon Wetzstein（斯坦福教授，团队页列为 Scientific Advisor / 联创） | 官网新闻稿、团队页 |
| 融资 | **4.5 亿美元 Series A**，投资方含 Capricorn、Khosla Ventures、Leitmotif、Matter Venture Partners、Mayfield、Premji Invest、Prelude Ventures、Temasek、Xora 等 | 官网新闻稿（公司自报） |
| 估值 | 官网新闻页转载的 Bloomberg 标题称 17 亿美元；属媒体口径，未读正文 | 媒体 |
| 产品 | FutureVision（智能层基础模型，预期未来授权给软硬件合作伙伴）；定制执行器机器人平台，额定负载 25 kg、峰值 40 kg | 官网首页、新闻稿 |
| 场景 | 制造与物流（汽车、电商等）；新闻稿称一次高产量制造评估中单轮不到两分钟、无人干预，超过客户 KPI | 官网新闻稿（自报） |

## 官方研究时间线（2026-10-16 共 3 篇）

完整核查见 [官网研究页归档](../../sources/sites/rhoda-ai.md)。

| 日期 | 研究 | 要点 | 本库入口 |
|------|------|------|----------|
| 2026-03-10 | [Causal Video Models Are Data-Efficient Robot Policy Learners](https://www.rhoda.ai/research/direct-video-action) | DVA：因果视频模型 + 逆动力学模型；约 10 小时数据即可学新任务；长上下文记忆与一次示范模仿 | [DVA](./rhoda-direct-video-action.md) |
| 2026-09-10 | [Does Scaling Web-Video Pre-training Help Real Robots Do Real Work?](https://www.rhoda.ai/research/scaling-web-video-pretraining) | 模型规模 XS→L：3.7%→84.7%；预训练算力与 DINO FD 同向 | [网络视频预训练 scaling](./rhoda-scaling-web-video-pretraining.md) |
| 2026-10-13 | [Giving Robots a Sense of Touch](https://www.rhoda.ai/research/giving-robots-a-sense-of-touch) | 薄型触觉传感器 + 多模态 DVA；盲操作取回纹理球 | [触觉 + DVA](./rhoda-dva-tactile-sensing.md) |

## 核心原理（对外可核对部分）

```mermaid
flowchart LR
  WEB["数亿条网络视频<br/>无动作标注"] --> PRE["原生因果视频预训练"]
  PRE --> POST["少量机器人数据后训练<br/>约 10–20 小时"]
  POST --> VID["预测短时未来画面"]
  OBS["相机 / 本体 / 语言"] --> VID
  VID --> IDM["逆动力学模型<br/>画面 → 末端动作"]
  IDM --> ACT["执行动作"]
  ACT --> OBS
```

- **预训练：** 无动作标注的网络视频，目标是学到运动、物理与交互的先验。
- **后训练：** 在具体本体上学习视频预测，以及「预测画面 → 动作」的映射（由独立的逆动力学模型完成，约 10 小时本体数据即可训练）。
- **运行：** 每几百毫秒重复「观察 → 预测未来视频 → 转动作 → 执行」。

## 工程实践

| 场景 | 建议 |
|------|------|
| 写综述 / 路线对照 | 归入「世界与动作基础模型」视角，见[公司路线对照](../comparisons/robot-foundation-model-company-paths-2026.md)；不要与开源 VLA 混排座次 |
| 评估 scaling 结论 | 引 [scaling 研究](./rhoda-scaling-web-video-pretraining.md) 时同时写明：单任务、单本体、每条件仅一次后训练、规模实验未做算力匹配 |
| 复现 | **不可复现**：官网未列代码、权重、数据、参数量；可复现的同类开源线见 [mimic-video](../methods/mimic-video.md) 与 [逆动力学模型](../concepts/inverse-dynamics-model.md) |
| 开源状态 | **确认未开源**；**不适用源码运行时序图** |

## 局限与风险

- 所有成功率、时长、数据量均为**公司自报**；DVA 博文没有成功率与基线对比，触觉博文没有成功率与试验次数。
- 成立时间官方未写；17 亿美元估值只见媒体标题。
- 模型参数量、训练数据规模、客户名称、传感器价格与规格未公开。
- 逆动力学模型按**本体类型**采集数据，并非完全通用（DVA 博文自述）。

## 关联页面

- [DVA：因果视频模型作机器人策略](./rhoda-direct-video-action.md)
- [网络视频预训练 scaling](./rhoda-scaling-web-video-pretraining.md)
- [触觉 + DVA](./rhoda-dva-tactile-sensing.md)
- [世界动作模型](../concepts/world-action-models.md)
- [逆动力学模型](../concepts/inverse-dynamics-model.md)
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)
- [Generalist AI](./generalist-ai-robotics.md) — 另一条闭源的 scaling 叙事，数据来源不同
- [Dyna-2](./dyna-2.md) — 另一条闭源世界动作模型线
- [公司技术路线对照（2026）](../comparisons/robot-foundation-model-company-paths-2026.md)

## 参考来源

- [Rhoda AI 官网研究页、新闻稿与团队页核查（2026-10-16）](../../sources/sites/rhoda-ai.md)
- 官网研究页：<https://www.rhoda.ai/research>
- 新闻稿：<https://www.rhoda.ai/news/press-release>

## 推荐继续阅读

- [Rhoda AI Research](https://www.rhoda.ai/research) — 官方研究博文列表
- [Rhoda AI 新闻稿：Exits Stealth with $450 Million Series A](https://www.rhoda.ai/news/press-release) — 公司定位、融资与 FutureVision 叙事
