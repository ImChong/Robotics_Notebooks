---
type: concept
tags:
  - world-models
  - taxonomy
  - pomdp
  - spatial-intelligence
  - world-labs
  - shanghai-ai-lab
status: complete
updated: 2026-10-15
summary: "Fei-Fei / World Labs 把过载的「世界模型」按 POMDP 闭环拆成 Renderer（输出观测）、Simulator（输出状态）、Planner（输出动作）；仿真是枢纽。上海人工智能实验室补上表征轴，并指出功能分类只描述解码、不定义内部压缩。"
related:
  - ../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md
  - ../methods/generative-world-models.md
  - ./world-action-models.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md
  - ../entities/world-labs.md
  - ../entities/marble-world-model.md
  - ./video-as-simulation.md
  - ../entities/cosmos-3.md
  - ../methods/model-based-rl.md
  - ../methods/vla.md
  - ../overview/gwm-closed-loop-5-papers-technology-map.md
  - ../entities/paper-gwm-first-principles.md
  - ../entities/paper-foredrive.md
  - ../entities/paper-worldline-action-driven-visual-simulation.md
sources:
  - ../../sources/blogs/worldlabs_functional_taxonomy_world_models.md
  - ../../sources/blogs/wechat_embodied_station_gwm_closed_loop_2026-09-10.md
  - ../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md
  - ../../sources/blogs/worldlabs_marble_world_model.md
  - ../../sources/papers/foredrive_arxiv_2609_26299.md
  - ../../sources/papers/worldline_arxiv_2609_38059.md
  - ../../sources/sites/worldline-project.md
  - ../../sources/repos/worldline.md
---

# 世界模型功能分类（Renderer / Simulator / Planner）

**世界模型功能分类** 是 Fei-Fei Li 与 World Labs（2026-06）提出的消歧：不按架构家族点名，而按系统在 **agent–环境环** 里 **输出哪一段**——像素观测、可计算状态，还是动作。

## 一句话定义

> **Renderer 给人看世界，Simulator 给人与程序算世界，Planner 让 agent 在世界里做事；三者是同一条 POMDP 环的投影，不是三种互斥产品。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| POMDP | Partially Observable Markov Decision Process | 分类所依附的闭环：动作改状态，agent 只见观测 |
| WM | World Model | 过载词；本页按输出功能拆，不按品牌拆 |
| WAM | World Action Model | Planner 侧、且通常横跨 Simulator 的联合建模范式 |
| 