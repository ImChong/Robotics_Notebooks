---
type: entity
tags: [paper, robotics, robot-learning]
status: complete
updated: 2026-10-07
project_id: dymd
arxiv: "2609.31349"
related:
  - ../overview/humanoid-motion-intelligence-day5-world-models-decision.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md
summary: "DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。"
---

# DyMD: Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models

DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉与指令条件化的机器人动作策略。 |
| WM | World Model | 预测行动后的环境变化。 |
| WAM | World Action Model | 联合建模未来视觉和动作。 |

## 为什么重要

DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。 评读时应将问题、观测条件与执行接口一并记录，避免从单个演示或指标外推能力。

## 流程总览

```mermaid
flowchart TD
    A["PF-Wan 14B 教师<br/>剖析 32 条轨迹的速度转向率 κ(σ)"] --> B["教师速度转向先验 νκ(t)<br/>强调去噪方向重组的噪声区间"]
    C["少步学生生成器<br/>(噪声 + 文本 + 首帧) → 学生 rollout x0"] --> D["解码视频 vs 配对目标视频<br/>冻结 V-JEPA 2.1 特征"]
    D --> E["时间亲和度 Ai<br/>衡量交互保真度"]
    E --> F["混合权重 gi = α / (α + (1-α)Ai)"]
    B --> G["自适应混合 νi = (1-gi)·νbase + gi·νκ<br/>采样 re-noise 时间步 ti"]
    F --> G
    C --> H["按 ti 对 rollout 加噪 → xt"]
    G --> H
    H --> I["冻结教师分数 sT 与 critic 分数 sφ<br/>DMD 分数差更新生成器"]
    C --> J["VAE 潜变量时间描述符<br/>+ 噪声嵌入 → 难度预测器 f"]
    J --> K["预测噪声相对拟合难度<br/>权重 wi = exp(β r̂i) 归一化至均值 1"]
    K --> L["加权 FM 损失更新 critic<br/>(与生成器共用 ti 采样规则)"]
    L --> I
```

图依据：论文 §3.1 至 §3.4 / Fig.4

## 方法栈

用分布匹配蒸馏将14B视频扩散教师压缩为1.3B少步生成器，并提高困难交互样本权重。

## 实验与评测

文章报告R-Bench遵循分数34.6升至44.2，另两套视频基准的领域分数提高；没有实体机器人试验。

## 与其他工作对比

它优化预测模型的成本与动态保真，不增加运行时控制模块；生成视频分数不是实机策略成功率。

## 工程实践

当前归档仅核实到arXiv入口，未核实可运行官方代码。

## 源码运行时序图

**不适用**：当前没有核实到该论文的可运行官方训练、推理或部署入口。

## 结论

**一句话总判：** DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。

1. 核对方法实际改变的是数据、表征、规划、控制还是评测。
2. 只把论文覆盖的任务、平台与扰动范围当作证据。
3. 复现前确认代码、权重、数据许可和硬件依赖状态。

## 局限与风险

它优化预测模型的成本与动态保真，不增加运行时控制模块；生成视频分数不是实机策略成功率。 当前导读以用户提供的文章和预印本题录为索引；详细配置与结果应核对论文全文。

## 关联页面

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [Day 5 原文归档](../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md)
- [arXiv:2609.31349](https://arxiv.org/abs/2609.31349)
- [arXiv PDF](https://arxiv.org/pdf/2609.31349)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1 至 3.4）、§4.1 实验设置；附录未读

## 推荐继续阅读

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
