---
type: tech_map_node
status: complete
topic: [imitation-learning, vla]
summary: "人形策略网络架构技术栈模块：梳理从浅层 MLP（DeepMimic/AMP）到 MoE、Diffusion Action Chunk、VLA 与 WAM 的骨干演化及真机部署取舍。"
sources:
  - ../../../sources/papers/deepmimic_reading_arxiv_1804_02717_dm1.md
  - ../../../sources/papers/amp_reading_arxiv_2104_02180_r7.md
  - ../../../sources/papers/ai_architecture_foundations.md
updated: 2026-11-24
---

# 人形策略网络架构

从浅层 MLP（DeepMimic / AMP）到 Diffusion chunk、VLA 与 WAM 的策略骨干演化；强调论文 Method 常见披露项与「小 MLP + 强系统」的真机现实。

## 在全栈中的位置

```mermaid
flowchart TD
    IN["观测输入：本体状态 + 历史滑窗 / 视觉深度"] --> BACK{"骨干形态与任务频段"}
    BACK -->|"50–500 Hz 运控闭环"| MLP["浅层 MLP / RNN / MoE（DeepMimic / AMP）"]
    BACK -->|"5–30 Hz 多模态操作"| DIFF["Diffusion Action Chunk / Flow Matching"]
    BACK -->|"高层语义与跨任务泛化"| VLA["VLA / WAM + 低层频率解耦控制"]
    MLP --> DEP["真机部署：推理延迟与控制频率对齐"]
    DIFF --> DEP
    VLA --> DEP
```

## 核心内容与站内入口

- **深度归纳**：[人形与腿式策略的网络架构](../../../wiki/concepts/humanoid-policy-network-architecture.md)
- **基础架构全景**：[AI 架构地图](../../../wiki/overview/ai-architecture-map.md) · [人形策略观测输入](../../../wiki/concepts/humanoid-policy-observation-inputs.md)
- **部署频率取舍**：[控制频率与推理频率解耦](../../../wiki/concepts/control-inference-frequency-decoupling.md)

## 上下游模块

- **上游与同层**：[Behavior Cloning](behavior-cloning.md) · [Diffusion Policy](diffusion-policy.md) · [PPO](../rl/ppo.md) · [全栈技术域总览](../../overview.md)
- **下游落地**：[人形强化学习](../rl/humanoid-rl.md) · [部署（上真机）](../system/deployment.md)

## 参考来源与收录边界

- [DeepMimic 通读归档（arXiv:1804.02717）](../../../sources/papers/deepmimic_reading_arxiv_1804_02717_dm1.md)
- [AMP 通读归档（arXiv:2104.02180）](../../../sources/papers/amp_reading_arxiv_2104_02180_r7.md)
- [AI 架构地图一手论文簇](../../../sources/papers/ai_architecture_foundations.md)
- 各类网络骨干的详尽参数对比见 [人形与腿式策略的网络架构](../../../wiki/concepts/humanoid-policy-network-architecture.md)。
