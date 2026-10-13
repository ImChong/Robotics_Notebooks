---
type: entity
tags: [foundation-policy, embodied-data, dexterous-manipulation, imitation-learning, scaling-laws]
status: active
updated: 2026-10-13
project_id: naturewill-natura-1
project: https://www.naturewill.com/blog/natura-1
related:
  - ../concepts/embodied-scaling-laws.md
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
  - ../concepts/robot-data-supervision-signal-types.md
  - ../methods/imitation-learning.md
  - ../methods/egoscale.md
  - ./dyna-2.md
sources:
  - ../../sources/sites/naturewill-natura-1.md
summary: "Nature Will 将 Natura-1 描述为从百万小时级第一人称人类操作经验预训练、再经少量真机数据适配的 50B MoE 物理世界模型，面向 20-DoF 灵巧手；官方公开研究预览，未提供代码、权重或数据下载。"
---

# Natura-1（Nature Will）

## 一句话定义

**Natura-1** 是 Nature Will 宣布的物理世界基础模型：先从大规模人类第一人称操作视频学习，再用有限真实机器人数据适配到双臂灵巧操作，目标是让同一策略根据语言指令操纵多自由度手部完成未见任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoE | Mixture of Experts | 专家混合网络；由多个专家子网络及路由机制组成 |
| DoF | Degrees of Freedom | 自由度；此处官方称目标灵巧手为 20-DoF |
| egocentric | Egocentric View | 第一人称视角；此处指人类自身视角操作视频 |
| VLA | Vision-Language-Action | 视觉-语言-动作范式；Natura-1 输入视觉和语言并输出机器人动作 |
| MPJPE | Mean Per-Joint Position Error | 平均关节位置误差；官方用于手部动作重建评估 |
| MFU | Model FLOPs Utilization | 模型浮点运算利用率；反映训练期间硬件算力利用情况 |

## 流程总览

Nature Will 披露的路线是“人类视频规模化预训练 + 少量真机适配”，并非只靠百万小时公开视频直接零样本控制机器人：

```mermaid
flowchart LR
  H["第一人称人类操作视频"] --> C["清洗、去重与分布控制"]
  C --> P["重建 3D 手姿并生成语言条件轨迹"]
  P --> M["50B MoE 视觉-动作预训练"]
  R["有限真实机器人操作数据"] --> A["灵巧手后训练适配"]
  M --> A
  A --> I["视觉与语言指令输入"]
  I --> D["双臂 20-DoF 灵巧手控制"]
  D --> E["真机操作与泛化评测"]
```

## 关键设计与官方披露

### 人类经验到动作监督

项目页称其内部数据基础设施处理数十 PB 级数据，并以每周数十万小时的吞吐处理第一人称视频。数据清洗包含视频/片段去重、视觉与传感异常筛查、任务配额控制；随后从单目和双目视频恢复稠密 3D 手部姿态，并把活动按总体目标、子任务和步骤进行层级标注。最终筛选语料被描述为百万小时级语言条件动作轨迹。

这条路线的关键是把观察到的人手运动转换为机器可学习的动作监督，再通过拟人化灵巧手与有限机器人数据完成形态适配。它不意味着人手关节轨迹可以不经映射直接下发给机器人；手部形态、接触动力学、传感器和控制接口仍构成显著的跨具身差异。

### 模型与执行栈

官方称 Natura-1 为 500 亿参数 MoE，联合视觉世界建模与动作生成，输入语言和视觉，在 20-DoF 双臂灵巧手平台运行。文章描述的内部实现包括分布式训练引擎、定制 attention kernel，以及推理侧 Rust 编排器和融合 CUDA kernel；作者报告真机观测到动作维持在 100 ms 闭环预算内。由于没有公开代码、配置或模型权重，这些是项目方披露的系统概况，不是可直接复现的部署配方。

## 评测读法

官方文章描述了 12 个灵巧操作能力维度、数分钟级多阶段任务和未见自然语言指令测试；平台包括 ROKAE AR5 双臂搭配 Wuji 灵巧手，以及 ARX AC One 双臂夹爪。文章称部分长时序任务使用额外后训练，其余演示共享同一组权重，因此阅读视频时要区分基座结果与额外调优结果。

官方还报告，预训练数据规模从约 4.4 万小时增加到 100 万小时，held-out 人类数据上的动作 MSE/MAE 分别下降 47.7%/37.3%，机器人验证数据上的预测 MSE/MAE 分别下降 67.7%/49.3%。这是基于 teacher-forcing 的动作预测误差，不是开放环任务成功率；还需关注预测设置、机器人验证集规模与构成，以及数据小时数之外的训练算力和模型配方。

基准比较不具备完全同构的平台条件：官方指出当前没有公认的灵巧手基础模型可直接作为严格同类基线；π0.5 和 GPT-6-Astra 在 ROKAE/Wuji 灵巧手平台无法产出可评分结果，另在 ARX 夹爪平台评估部分基线。跨末端执行器的分数不能简单解释为算法优劣；需共同查看硬件、训练数据、调优预算、任务定义和成功判定。

## 对研究与工程的意义

- **数据规模之外还要有监督密度。** 人类视频只有经过手姿重建、语言标注、任务分层和质量控制，才可能转为动作学习信号；小时数本身不是充分的数据质量指标。
- **跨具身迁移仍需要真机适配。** Nature Will 明确描述了有限 robot data post-training，因此 Natura-1 不是完全不需要机器人数据的纯人类视频零样本迁移案例。
- **闭环延迟是系统指标。** 100 ms 是官方描述的端到端预算，工程复现时应拆分视觉采集、模型推理、通信、动作下发和控制器周期，避免仅以模型单次推理时延代替闭环延迟。
- **大模型演示不等于通用可靠性。** 公开页面是项目方研究预览，任务选择、基准实现、失败率与独立复现实验仍需持续核验。

## 开源状态与局限

| 资源 | 状态（截至 2026-10-13） |
|------|------------------------|
| 研究介绍与媒体演示 | 官方项目页公开 |
| 训练/推理/机器人部署代码 | 页面未提供公开仓库链接 |
| Natura-1 权重 | 页面未提供下载入口 |
| 百万小时人类经验数据 | 页面未提供数据集下载入口；属内部整理语料的披露 |
| 可复现训练配方与完整评测集 | 页面未提供 |

因此目前应按**闭源研究预览**阅读。百万小时数据规模、模型参数量、基准结果和闭环时延均来自项目方披露；外部读者暂不能独立复现或完整审计其数据来源、许可、标注误差与实验协议。未来若官方发布论文、代码或数据，应重新核验并更新本页。

## 关联页面

- [具身规模法则](../concepts/embodied-scaling-laws.md) — 人类经验小时数与预测/下游能力缩放的对照案例。
- [具身数据采集四层分类](../concepts/embodied-data-collection-four-layers-taxonomy.md) — 区分观察、动作、状态和结果等数据监督层级。
- [机器人数据监督信号类型](../concepts/robot-data-supervision-signal-types.md) — 理解视频、动作轨迹与语义标注各自提供的监督。
- [模仿学习](../methods/imitation-learning.md) — 从人类/专家演示训练机器人策略的基础范式。
- [EgoScale](../methods/egoscale.md) 与 [Dyna-2](./dyna-2.md) — 对照研究性缩放分析与产业方百万小时预训练主张。

## 参考来源

- [Natura-1 官方研究预览归档](../../sources/sites/naturewill-natura-1.md)

## 推荐继续阅读

- [Nature Will: The Will to Act — Natura-1](https://www.naturewill.com/blog/natura-1)
- [具身规模法则](../concepts/embodied-scaling-laws.md)
- [EgoScale：Scaling Egocentric Video for Humanoid Robot Learning](../methods/egoscale.md)
