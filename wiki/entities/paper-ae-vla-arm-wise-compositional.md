---
type: entity
project_id: ae-vla-acg-bench
tags: [paper, manipulation, bimanual, vla, compositional-generalization, lora, benchmark, pi05, so101]
status: complete
updated: 2026-10-12
arxiv: "2610.06184"
venue: "arXiv 2026 (Technical Report)"
summary: "ACG-Bench（8 任务族、23 条件）+ AE-VLA（臂 token 分组 + SkillLoRA + 臂间注意力阻断，建于 π0.5）：未见组合泛化 CCSR 21.53% 对单 π0.5 的 2.94%，SO101 双臂真机五个未见条件 39% 对最强基线 10%。"
related:
  - ../overview/frontier-manipulation-2026-10-05-10-09.md
  - ../tasks/bimanual-manipulation.md
  - ../tasks/manipulation.md
  - ../concepts/lora.md
  - ../concepts/mixture-of-experts.md
  - ./paper-pi05-open-world-vla.md
  - ./paper-core-collaboration-role-experts.md
sources:
  - ../../sources/papers/ae_vla_acg_bench_arxiv_2610_06184.md
  - ../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md
---

# AE-VLA 与 ACG-Bench

**AE-VLA / ACG-Bench**（arXiv:2610.06184，Zhang 等，通讯方向含 Huchuan Lu）把“双臂已分别学会单臂技能，能否按臂重新组合”定义为 **arm-wise compositional generalization**，提供 ACG-Bench 基准，并在 π0.5 上用**臂 token 分组、SkillLoRA、Arm-wise Attention** 组合出 AE-VLA。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACG | Arm-wise Compositional Generalization | 按臂重组已学技能的泛化 |
| AE-VLA | Token Group + SkillLoRA + AWA 的 VLA | 本文最终模型 |
| AWA | Arm-wise Attention | 阻断跨臂直接动作注意力 |
| CCSR | Constraint-Compliant Success Rate | 需满足目标、里程碑、顺序、时序的成功率 |
| LoRA | Low-Rank Adaptation | 低秩适配器，SkillLoRA 按技能类型索引 |
| MA-VLA | Multi-Arm VLA baseline | Arm Shuffle 增广的共享模型基线 |

## 为什么重要

- 现有双臂基准多测“同任务”成功率；本文测重排序、同步与跨任务重组，直指技能复用。
- 数字暴露差距：单 π0.5 域内 37.00%，未见组合只有 2.94%。

## 方法栈与流程总览

```mermaid
flowchart LR
  I["图像+指令+每臂技能提示"] --> TG["Token Group: 每臂 H=50 动作 token, 共享动作专家"]
  TG --> SL["SkillLoRA: 10 个 rank-4 适配器, 每臂路由器硬选"]
  SL --> AW["AWA: 阻断跨臂动作注意力, 块内下三角"]
  AW --> FM["流匹配 10 步 Euler 输出双臂动作"]
```

## 方法

- **ACG-Bench**：ManiSkill + RoboTwin 2.0 管线；6 个源任务族（Stack Bowls、Stack Cubes、Push Cubes、Burger & Fries、Can in Basket、Object in Cabinet）+ 2 个留出跨任务族（Cube in Bowl、Bowl in Cabinet）；6 个域内 + 17 个未见（4 Reorder、6 Sync、4 Sync+Reorder、3 Cross-task）；训练 600 条示范（每源任务 100），每条件评测 100 回合共 2,300 回合。
- **基线**：Single π0.5、Dual（两个独立单臂策略）、MA-VLA（Arm Shuffle）。
- **AE-VLA 组件**：Token Group（固定 token 区间识别手臂，非学习嵌入）；SkillLoRA（10 个 rank-4，α=16，路由损失权重 0.1，覆盖 18 个动作专家块）；AWA（各动作块下三角，保留全局上下文）。
- **训练**：30,000 步，批 32，2×A800，AdamW，峰值 lr 5e-5。

## 实验与评测

- **仿真未见 CCSR**：Single 2.94，MA-VLA 3.06，Dual 5.53，AE-VLA 21.53；分组 Reorder 26.50、Sync+Reorder 22.00、Cross-task 40.00、纯 Sync 8.67；域内 Single 37.00 vs AE-VLA 27.17。
- **消融**：Token Group 3.82 → +SkillLoRA 6.00 → +AWA 5.82 → 两者同加 21.53；AWA 使 Token Group 域内 45.00 → 25.00。
- **真机**：两台 SO101、1 全局 + 2 腕部相机，3 任务各 50 条 LeRobot 示范；3 域内 + 5 未见条件各 20 次。未见平均 Single 2.00%、MA-VLA 6.00%、Dual 10.00%、AE-VLA 39.00%；Stack Bowls/Sync AE-VLA 12/20，所有基线 0/20。域内 Dual 70.00% 高于 AE-VLA 60.00%。

## 源码运行时序图

不适用：论文与 arXiv 页未列出代码、数据或项目页（核查 2026-10-12），基准构建依赖 ManiSkill / RoboTwin 2.0 公开管线，但 ACG-Bench 本身未发布。

## 工程实践

- 评测提供每臂技能提示，系统上需要上游规划器给出“左臂做什么、右臂做什么”。
- 采用 CCSR 而非单一终态成功率，复现需同时记录里程碑与时序。
- 开源状态：**未发布**。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [π0.5](./paper-pi05-open-world-vla.md) | 本文的统一底座与 Single 基线 |
| [CoRE](./paper-core-collaboration-role-experts.md) | 同为按角色拆分共享参数，但 CoRE 是多机器人去中心化、路由无标签 |
| [LoRA](../concepts/lora.md) / [MoE](../concepts/mixture-of-experts.md) | SkillLoRA 是技能索引的硬路由 LoRA |
| TwinVLA / SkillVLA | 因预训练不同或实现不可用，未纳入定量比较 |

## 结论

**AE-VLA 证明“按臂结构化”能把未见组合从 2.94% 拉到 21.53%，但同步与域内性能是代价。**

1. **泛化与域内权衡** — AWA 使域内掉 20 个百分点。
2. **两部件缺一不可** — SkillLoRA 与 AWA 单加都只有 6.00 / 5.82。
3. **同步最弱** — 纯 Sync 仅 8.67%。
4. **真机提升更大但样本小** — 39% vs 10%，每条件 20 次。
5. **依赖外部技能标签** — 错误标签未测。

## 局限与风险

- SkillLoRA 收益混杂路由监督和额外参数，无参数匹配控制。
- Dual 的碰撞成功被人工记 0，Single/MA-VLA 日志重新评分；评测主观性需留意。
- 无代码、无项目页；机构单位未在页面中列出（未核实）。

## 关联页面

- [Manipulation 周报汇总（2026-10-05 至 10-09）](../overview/frontier-manipulation-2026-10-05-10-09.md)
- [双臂操作任务页](../tasks/bimanual-manipulation.md)
- [LoRA 概念页](../concepts/lora.md)
- [CoRE：协作角色专家](./paper-core-collaboration-role-experts.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页 ACG-Bench 属③策略任务成功率层：以 CCSR 衡量双臂按臂重组技能的未见组合泛化，是跨组合分档而非单任务均值

## 参考来源

- [ae_vla_acg_bench_arxiv_2610_06184.md](../../sources/papers/ae_vla_acg_bench_arxiv_2610_06184.md)
- [frontier_manipulation_2026_10_05_10_09.md](../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md) — 周报线索
- 论文：<https://arxiv.org/abs/2610.06184>

## 推荐继续阅读

- [RoboTwin 2.0](https://robotwin-platform.github.io/)
- [ManiSkill3](https://www.maniskill.ai/)
