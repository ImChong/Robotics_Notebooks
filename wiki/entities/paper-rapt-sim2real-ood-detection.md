---
type: entity
tags:
  - paper
  - humanoid
  - deployment
  - ood-detection
  - sim2real
status: complete
updated: 2026-09-20
arxiv: "2602.01515"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_37_rapt-sim2real-ood-detection.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "仿真学习标称执行流形，部署时用预测偏差做 OOD 检测与 sim2real 失配诊断。"
---

# RAPT: model-predictive out-of-distribution detection and failure diagnosis for sim-to-real humanoid deployment

**RAPT: model-predictive out-of-distribution detection and failure diagnosis for sim-to-real humanoid deployment**（[arXiv:2602.01515](https://arxiv.org/abs/2602.01515)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[37/44]**，归类 **监控与评测**。

## 一句话定义

仿真学习标称执行流形，部署时用预测偏差做 OOD 检测与 sim2real 失配诊断。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RAPT | Robust Adaptive Prediction for Transfer | 文内部署监控框架 |
| OOD | Out-of-Distribution | 分布外检测 |
| Sim2Real | Simulation to Real | 仿真到真机 |

## 为什么重要

- 文内部署期监控完整方案；89% TPR / 87.5% Top-1 诊断等数字出处。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **监控与评测** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | arXiv 2026 |
| **文内章节** | 监控与评测 |
| **要点** | 模型预测 + 失配度量 +（可选）LLM 根因推理。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**策略无自我评判机制，监控应作为部署标配而非附加项。**

1. 文内角色：监控与评测 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：模型预测 + 失配度量 +（可选）LLM 根因推理。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_37_rapt-sim2real-ood-detection.md](../../sources/papers/freedof_sim2real_37_rapt-sim2real-ood-detection.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2602.01515)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
