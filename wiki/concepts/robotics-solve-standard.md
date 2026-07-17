---
type: concept
tags: [evaluation, benchmarking, home-robot, deployment, reliability, sunday-robotics, solve-standard]
status: complete
updated: 2026-07-17
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../entities/sunday-robotics-act2.md
  - ../tasks/manipulation.md
  - ../entities/tidybot2.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/sunday_act2_preview.md
summary: "Solve（Sunday Robotics, 2026）将机器人能力进展解构为 Performance、Scope、Adaptation cost 三元组；同一成功率数字在不同边界下含义不同，旨在使家用机器人结果可比较、可累积。"
---

# Robotics Solve 标准

## 一句话定义

**Solve** 是 Sunday Robotics 在 2026 年提出的机器人能力 **声明格式**：在明确 **Scope（适用分布）** 与 **Adaptation cost（部署适配成本）** 的前提下报告 **Performance（成功/质量/速度）**，避免 demo 成功率脱离边界被误读。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Solve | （Sunday 术语，非通用缩写） | 可靠性能 + 声明边界 + 适配成本的三元组 |
| SR | Success Rate | 任务成功率，Solve 的 Performance 维度之一 |
| OOD | Out-of-Domain | Scope 外的环境/物体/布局 |
| SFT | Supervised Fine-Tuning | 一种 Adaptation cost（部署后微调） |

## 为什么重要

- **解构 headline 数字：** **99.1%** 在「单衣物单房间」与「9 类衣物 × 未见家庭 × 零适配」是 **不同科学主张**；Solve 强制读者同时看到三者。
- **对标航空史隐喻：** Sunday 用莱特兄弟 **12 秒首飞** vs **1914 定期航班** 类比 robotics 从 **可能性 demo** 到 **可重复服务** 的转折——评测单位应从「能否飞」转向「能否每日准时飞」。
- **与学术 benchmark 互补：** LIBERO、CALVIN 等固定 sim bench 擅长 **算法对比**；Solve 强调 **真实部署分布 + 适配经济学**，更接近产品/研究交界处的 **claim audit**。
- **首个实例：** [ACT-2 叠衣 Solve](../entities/sunday-robotics-act2.md) 声明 **零 per-home 适配**、**固定 checkpoint**、**785 次** 自主尝试与 **预锁定 rubric**。

## 核心结构

### 三元组

| 组件 | 回答的问题 | 典型声明示例（ACT-2 叠衣） |
|------|------------|---------------------------|
| **Performance** | 在边界内多好？ | SR **99.1%**；折痕 **4.72/5**；median **2:13** |
| **Scope** | 对什么分布负责？ | 9 类衣物、未见房间/床/光照/站位、篮中/堆/地面初始态 |
| **Adaptation cost** | 新部署要付什么？ | **零** home-specific 数据、示范、微调 |

三者 **缺一不可**：只报 Performance 而不报 Scope/Adaptation，在 Sunday 框架下 **不构成 Solve**。

### 与常见评测习惯对照

| 习惯 | 缺口 | Solve 补位 |
|------|------|------------|
| **单次 demo 视频** | 环境/干预/重置未声明 | Scope + 定量 Performance |
| **Sim benchmark 榜首** | 分布与真机部署差距大 | 强调 **真实家庭** 与 **零适配** |
| **Few-shot 微调 SOTA** | 隐藏 per-site 采集成本 | **Adaptation cost** 显性化 |
| **平均成功率** | 长尾衣物/布局被均值掩盖 | Scope 内 **分桶报告**（衣物型/初始态/站位） |

### 可信 Solve 的最低工程要求（归纳自 ACT-2 博文）

1. **Scope 与 rubric 评测前锁定**，事后不改标准。
2. **Adaptation 边界可审计**：例如评估 homes 是否进入 post-training / model selection。
3. **分桶样本量透明**：避免 n 过小分项过度解读。
4. **独立或双审标注**（ACT-2：双轮 annotator + review lead）。

## 常见误区或局限

- **Sunday 专有术语：** 尚未成为社区通用标准；跨论文对比仍需映射到 Scope/Adaptation 等价描述。
- **不替代 sim bench：** Solve 贵、慢、难复现；算法迭代仍依赖仿真与小规模真机。
- **Adaptation = 0 的审计难：** 外界难以验证数据隔离与 checkpoint 冻结。
- **Scope 排除项：** ACT-2 明确 **不** 覆盖袜/内衣等—— headline SR **不** 代表「全部洗衣」。

## 关联页面

- [ACT-2（Sunday Robotics）](../entities/sunday-robotics-act2.md) — 首个完整 Solve 实例与定量结果
- [Manipulation](../tasks/manipulation.md) — 家用可变形操作任务语境
- [TidyBot2](../entities/tidybot2.md) — 开源 household 平台（可作不同 Adaptation 假设下的对照）
- [VLA](../methods/vla.md) — 操作基础模型评测与 few-shot 叙事
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 其 ③ 策略成功率评测层的可比性基础：Solve 三元组回答「同一成功率数字在不同边界下是否可比、可累积」

## 推荐继续阅读

- Sunday 官方原文 § *Solve: A New Standard for Robotics Progress*：<https://www.sunday.ai/blog/act-2-preview>

## 参考来源

- [sunday_act2_preview.md](../../sources/blogs/sunday_act2_preview.md)
