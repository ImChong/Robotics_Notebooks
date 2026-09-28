---
title: "Physical RSI 1.0: Recursive Self-Harness for Scaling Embodied Skills"
source: "https://mmlab.hk/research/PhysicalRSI"
date: "2026-09-28"
note: "WebFetch 正文摘录；完整交互页含大量内联 SVG/视频 URL，未整页入库"
---

# Physical RSI 1.0

A Simple and Effective Baseline that Ranks No.1 on the Challenging RoboDojo Benchmark

- Official Overall · #1 Physical RSI — **36 Score**, **31% SR**（页面标注 2026-09-28 RoboDojo-Sim）
- 主测试床：RoboDojo（42 sim 五维任务生态）

## Embodied Self-Harness（页内公式）

- \(A_k = \mathrm{Agent}(F, H_k)\)
- \(H_{k+1} = \mathrm{Improve}(A_k, H_k, \tau_k)\)
- **F**：System 2 多模态 agent（理解、决策、改写 harness）
- **H_k**：可编辑 harness（路由、skills、tools、System 1 控制代码）
- **τ_k**：具身反馈（观测、动作、成败）

伪代码骨架（页内）：

```
H[0] = initial_harness()
for k in range(budget):
    A[k] = Agent(F, H[k])          # S2 + H
    plan = A[k].understand(task)    # S2
    τ[k] = H[k].execute(plan)       # S1
    variants = A[k].rewrite_many(H[k], τ[k])  # S2 → H
    pool = [Agent(F, h) for h in variants]
    survivor = select_on_eval([A[k], *pool])
    H[k+1] = survivor.harness
```

## 技能与工具

- System 1 可组合：**code-policy skills**、**π₀.₅**（VLA motor tool）、稀疏 memory 等
- 技能库展示：**212** clips，**40** tasks

## 外部链接（页内可点击）

- https://mmlab.hk/
- https://www.kinetixai.tech/
- https://robodojo-benchmark.com/leaderboard
