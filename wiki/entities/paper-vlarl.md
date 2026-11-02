---
type: entity
tags: [paper, robotics, robot-learning]
status: complete
updated: 2026-10-07
project_id: vlarl
arxiv: "2609.30868"
related:
  - ../overview/humanoid-motion-intelligence-day5-world-models-decision.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md
summary: "VLaRL冻结基础VLA，在其动作上叠加由仿真训练的潜变量条件残差策略。"
---

# VLaRL: Augmenting Vision-Language-Action Models with Simulation-Trained Latent-Conditioned Residual RL

VLaRL冻结基础VLA，在其动作上叠加由仿真训练的潜变量条件残差策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉与指令条件化的机器人动作策略。 |
| WM | World Model | 预测行动后的环境变化。 |
| WAM | World Action Model | 联合建模未来视觉和动作。 |

## 为什么重要

VLaRL冻结基础VLA，在其动作上叠加由仿真训练的潜变量条件残差策略。 评读时应将问题、观测条件与执行接口一并记录，避免从单个演示或指标外推能力。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["Step 1：微调并冻结 VLA"]
        A["真机示教（每任务 32 条）"] --> B["微调 VLA 后冻结"]
        A --> C["重建数字孪生仿真环境<br/>（FoundationPose 重建物体位姿）"]
    end
    subgraph S2["Step 2：学习潜变量映射"]
        D["仿真观测 → VLM 得 Z_s"] --> E["Mapper M_θ（Transformer）<br/>Sinkhorn 最优传输对齐到真实分布"]
        F["真实观测 → VLM 得 Z_r"] --> E
    end
    subgraph S3["Step 3：仿真中训练残差策略（VLA 与 Mapper 冻结）"]
        G["映射后的 Z̃_s"] --> H["冻结 VLA 动作头 → 名义动作 a_VLA"]
        G --> I["MeanPool 得 z̃_s"]
        H --> J["残差策略 TD3<br/>输入 z + a_VLA + 本体状态 + 腕部力"]
        I --> J
        J --> K["a = a_VLA + α·a_RL"]
    end
    subgraph S4["Step 4：真机部署（无 Mapper、无真机 RL）"]
        L["真实观测 + 语言指令 → VLM 得 Z_r"] --> M["冻结 VLA → a_VLA；MeanPool 得 z_r"]
        M --> N["残差策略 → a = a_VLA + α·a_RL"]
    end
    B --> D
    B --> F
    C --> D
    E --> G
    J -. "训练好的残差策略" .-> N
```

图依据：论文 §III（Overview、Step 1–4）/ Fig.2。

## 方法栈

残差策略读取视觉、本体和力反馈，潜变量映射对齐仿真与真机表征；真机执行时修正动作而不在线训练。

## 实验与评测

文章报告八种真机任务组合改善；如按钮任务67.5%升至100%，叠杯17.5%升至45%。

## 与其他工作对比

相较端到端重训，模块化复用基础策略；迁移效果依赖仿真接触与本体覆盖。

## 工程实践

当前归档仅核实到arXiv入口，未核实可运行官方代码。

## 源码运行时序图

**不适用**：当前没有核实到该论文的可运行官方训练、推理或部署入口。

## 结论

**一句话总判：** VLaRL冻结基础VLA，在其动作上叠加由仿真训练的潜变量条件残差策略。

1. 核对方法实际改变的是数据、表征、规划、控制还是评测。
2. 只把论文覆盖的任务、平台与扰动范围当作证据。
3. 复现前确认代码、权重、数据许可和硬件依赖状态。

## 局限与风险

相较端到端重训，模块化复用基础策略；迁移效果依赖仿真接触与本体覆盖。 当前导读以用户提供的文章和预印本题录为索引；详细配置与结果应核对论文全文。

## 关联页面

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [Day 5 原文归档](../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md)
- [arXiv:2609.30868](https://arxiv.org/abs/2609.30868)
- [arXiv PDF](https://arxiv.org/pdf/2609.30868)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 方法（A–E）、§IV 实验设定、§V 结果开头

## 推荐继续阅读

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
