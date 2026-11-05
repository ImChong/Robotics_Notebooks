---
type: entity
tags:
  - paper
  - vla
status: complete
updated: 2026-09-26
arxiv: "2602.09430"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md
sources:
  - ../../sources/papers/atombridge_arxiv_2602_09430.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md
summary: "AtomBridge（arXiv:2602.09430）：长程科学实验由多个原子任务串联，而在原子任务上微调的 VLA 前一任务的终止状态可能落在后一任务的有效初始状态分布之外（skill-chaining gap）；AtomBridge 是推理时外挂、VLA 权重冻结的 LLM agent 插件：Intervention Switch（冻结 CLIP）判定任务完成，ready-state 检索取下一任务的 pre-contact 状态，Planning Agent + Coding Agent 生成过渡动作代码，无需额外 VLA 微调或长程演示（依据 arXiv:2602.09430v2 摘要、§3）。…"
---

# AtomBridge（arXiv:2602.09430）

**AtomBridge**（[arXiv:2602.09430](https://arxiv.org/abs/2602.09430)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第三篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) **架构/异步** 段。

## 一句话定义

**长程科学实验由多个原子任务串联，而在原子任务上微调的 VLA 前一任务的终止状态可能落在后一任务的有效初始状态分布之外（skill-chaining gap）；AtomBridge 是推理时外挂、VLA 权重冻结的 LLM agent 插件：Intervention Switch（冻结 CLIP）判定任务完成，ready-state 检索取下一任务的 pre-contact 状态，Planning Agent + Coding Agent 生成过渡动作代码，无需额外 VLA 微调或长程演示（依据 arXiv:2602.09430v2 摘要、§3）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| GRPO | Group Relative Policy Optimization | 组内相对策略优化（FIRE-VLA / Temporal GRPO 语境） |
| RL | Reinforcement Learning | 强化学习后训练 |

## 为什么重要

- 在原子任务上微调的 VLA 串联执行长程任务时存在 robot-state mismatch 导致的 skill-chaining gap；AtomBridge 在任务边界用 LLM agent 推理 + 机器人动作代码生成插入过渡动作，VLA 权重固定。v2 摘要：8 步组合任务全序列成功率提升 10%~25%（依据 arXiv:2602.09430v2 摘要）。
- 策展机构：东南大学、同济大学
- 与 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md) 同批工作可横向对照。

## 流程总览

```mermaid
flowchart TD
  A["固定的原子任务 VLA 策略 τ_i<br/>（π0 / π0.5，不重训）"]
  SW["Intervention Switch<br/>冻结 CLIP 图文编码器 → 完成概率 p_t<br/>连续 w 次 ≥ η 或 t ≥ H_i 触发"]
  PAUSE["暂停当前策略"]
  MEM[("原子任务 memory M<br/>每任务一条离线演示")]
  RET["ready-state 检索<br/>视觉 judge + 滑窗搜索 → s_ready"]
  PLAN["Planning Agent<br/>输入 s_cur、s_ready、标定相机视图 V、参数 κ"]
  TOOL["Tool Box：检索 / 碰撞检查 / 代码检查"]
  CODE["Coding Agent<br/>生成模板化 Python 过渡动作代码"]
  CHK["固定指令集校验<br/>schema / 限幅 / 仅过渡 / 碰撞"]
  EXE["执行过渡动作 a_tr<br/>到达下一任务 pre-contact 状态"]
  R["重置历史，恢复 VLA 执行 τ_{i+1}"]
  A --> SW --> PAUSE --> PLAN
  MEM --> RET --> PLAN
  PLAN <--> TOOL
  PLAN --> CODE --> CHK --> EXE --> R
  R -. 循环：完成→过渡→恢复 .-> A
```

图依据：论文 §3、§3.1–3.2、Fig. 2

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2602.09430](https://arxiv.org/abs/2602.09430) |
| **文内评测** | MuJoCo 仿真（Autobio 3D 资产）+ 真实热循环仪场景（Task G）；基座 π0 / π0.5，agent 底座 Qwen3.5-397B-A17B；19 个原子任务各 100 条演示；基线 Vanilla / MP（RRT-Connect）全序列均 0/20，AtomBridge 在 11/12 配置非零（2/20~14/20），真机 Task G 4/10 对 0/10（依据 v2 §4.1、§4 结果、Tab. 1–2） |
| **开源** | **待核实**（2026-09-26） |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。

## 实验与评测

- **文内口径：** MuJoCo + Autobio 仿真 + 真机热循环仪（v2 §4）
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Temporal GRPO](./paper-temporal-grpo.md) | 同周「训练范式」线：阶段信用 vs 失败驱动 teacher（若适用） |
| [VLA 方法页](../methods/vla.md) | 本页为单篇索引；机制细节以原文为准 |

## 结论

**AtomBridge 适合作为本期「架构/异步」路线的快速索引页。**

1. 核心贡献：在原子任务上微调的 VLA 串联执行长程任务时存在 robot-state mismatch 导致的 skill-chaining gap；AtomBridge 在任务边界用 LLM agent 推理 + 机器人动作代码生成插入过渡动作，VLA 权重固定。v2 摘要：8 步组合任务全序列成功率提升 10%~25%（依据 arXiv:2602.09430v2 摘要）。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第三篇）](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part3.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)
- [arXiv:2602.09430](https://arxiv.org/abs/2602.09430)
- 核对（2026-11-02）：对照 arXiv:2602.09430v2（14 Aug 2026）摘要、§3、§4、结论，删去与论文不符的「Haar 子空间 / Flow-DPO / 动作骨架与残差 / 异步换指令」描述，改为 LLM agent 推理插件；MuJooco 笔误改 MuJoCo；流程图与论文一致，未改。
- [arXiv PDF](https://arxiv.org/pdf/2602.09430)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§2 Related Work、§3 AtomBridge（含 3.1 Intervention Switch、3.2 Transition Generation）、§4.1 实验设定；§4 其余实验与补充材料仅略读/未读

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2602.09430)
