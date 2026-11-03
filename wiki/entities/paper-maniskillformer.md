---
type: entity
tags:
  - paper
  - manipulation
  - bimanual
  - llm
  - compositional
status: complete
updated: 2026-09-20
arxiv: "2609.16331"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/maniskillformer_arxiv_2609_16331.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md
summary: "ManiSkillFormer（arXiv:2609.16331）：Task-Conditioned Geometric Contract 声明 keypoint/normal 等 primitive；LLM 生成 contract+模板，视觉定位后实例化；Galaxea R1-Lite 无示范 Pick-"
---

# ManiSkillFormer（arXiv:2609.16331）

**ManiSkillFormer**（*ManiSkillFormer: Demonstration-Free Compositional Manipulation via Task-Conditioned Geometric Contracts*，[arXiv:2609.16331](https://arxiv.org/abs/2609.16331)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md)（2026-09-14–18）。

## 一句话定义

**Task-Conditioned Geometric Contract 声明 keypoint/normal 等 primitive；LLM 生成 contract+模板，视觉定位后实例化；Galaxea R1-Lite 无示范 Pick-and-Place 88.24%。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LLM | Large Language Model | 大语言模型 |
| BC | Behavior Cloning | 行为克隆 |
| EE | End-Effector | 末端执行器 |

## 为什么重要

- 无示范组合操作需可验证几何契约而非黑盒 policy。

## 流程总览

```mermaid
flowchart TD
    subgraph OFF["离线：人类专家定义 schema（§IV-A）"]
        A["原子技能 A、元技能 M、分解规则 D"]
        B["每个原子技能的 contract schema Σc（关键点语义标签）与 skill schema Σ（有序运动阶段、偏移与容差类别）"]
    end
    subgraph SEMI["半在线：规划与技能适配"]
        C["用户指令 l"] --> D["规则式 Planner：动词短语匹配技能词表 + 名词短语匹配物体<br/>输出技能-物体序列 p"]
        D --> E["元技能按分解规则展开为原子技能"]
        E --> F["Contract 代理（LLM）：填关键点自由文本描述 d 与表面法向约束 n"]
        E --> G["Skill 代理（LLM）：为各阶段选偏移距离类别与到达容差类别"]
        F --> H["已填 contract c"]
        G --> I["已填运动模板 μ（不含场景坐标）"]
        H --> J["原子技能执行程序 p_c"]
        I --> J
    end
    A --> E
    B --> F
    B --> G
    subgraph ON["在线：落地与执行"]
        K["多视角标定相机观测 o"] --> L["可提示 3D 感知：LLM 按 contract 预测各视角 2D 语义关键点"]
        J --> L
        L --> M["LFM 3D 提升 → 估计局部表面法向，检查是否落入 contract 允许范围"]
        M --> N["相机外参 / 正运动学 → 机器人基座系 3D 关键点 θ"]
        N --> O["按物体尺寸缩放偏移范围，采样偏移与容差 → 笛卡尔路点"]
        O --> P["控制器 C_r：按可达性分配手臂 → 逆运动学（含自碰撞避免）→ 关节轨迹 → PD 执行"]
    end
```

图依据：论文 §IV-A 至 §IV-F 与 Fig. 1 / Fig. 2

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.16331](https://arxiv.org/abs/2609.16331) |
| **开源** | **待发布**（步骤 2.5，2026-09-20） |
| **方法摘要** | LLM contracts + motion templates + 3D primitive grounding on dual-arm platform. |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。

## 实验与评测

- Pick-and-place 88.24% avg without per-object demos（作者报告）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**ManiSkillFormer 用几何契约 + LLM 模板实现无示范双臂组合操作。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [loco-manipulation](../tasks/loco-manipulation.md)

## 参考来源

- [maniskillformer_arxiv_2609_16331.md](../../sources/papers/maniskillformer_arxiv_2609_16331.md)
- [wechat_senlanke_weekly_manipulation_2026-09-14_18.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md)
- [arXiv:2609.16331](https://arxiv.org/abs/2609.16331)
- [arXiv PDF](https://arxiv.org/pdf/2609.16331)：已通读方法部分（实验与附录以略读为准）：§III 问题定义，§IV 方法（系统概览、Planner、几何 contract、3D 感知模块、AgenSkillGraph、控制器）

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.16331)
