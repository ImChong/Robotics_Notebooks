---
type: overview
tags: [humanoid, motion-control, trends, bfm, mpc, reinforcement-learning]
status: complete
updated: 2026-07-14
related:
  - ./humanoid-motion-control-know-how-technology-map.md
  - ./humanoid-rl-motion-control-body-system-stack.md
  - ../concepts/behavior-foundation-model.md
  - ../../roadmap/depth-bfm.md
sources:
  - ../../sources/papers/humanoid_motion_control_know_how.md
  - ../../sources/raw/feishu_humanoid_motion_control_know_how_2026-07-14.md
summary: "飞书 Know-How 开篇趋势判断：三年从 MPC 传统栈到 BFM 运控大模型；硬件/数据/算法/benchmark 四重挑战并存。"
---

# 人形机器人运动控制发展趋势

基于 [RoboParty 飞书 Know-How](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87) 开篇与 [2026-07 抓取摘录](../../sources/raw/feishu_humanoid_motion_control_know_how_2026-07-14.md) 的**趋势层**归纳：传统 MPC 栈与 BFM 范式快速演进，但具身智能仍受硬件形态、数据规模与算法范式、评测标准四重制约。

## 一句话定义

运控从「模型预测 + WBC」扩展到「数据驱动 + 身体基础模型」，但真机物理闭环仍是主战场。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 运控大模型 / 身体基座 |
| MPC | Model Predictive Control | 传统主干之一 |
| RL | Reinforcement Learning | 腿足成功关键催化剂 |
| Sim2Real | Simulation to Real | 数据与部署鸿沟 |
| VLA | Vision-Language-Action | 上层调用身体的远期接口 |
| IL | Imitation Learning | 数据筛选与质量争议焦点 |

## 为什么重要

- **路线选择**：判断项目应投 Model-based、RL 还是 BFM 预训练。
- **与姊妹地图互补**：[八层身体系统栈](./humanoid-rl-motion-control-body-system-stack.md) 偏论文栈；本页偏飞书作者对**产业节奏**的判断。
- **避免单线叙事**：文档明确传统控制仍是基本盘，BFM 是扩展而非完全替代。

## 核心判断（来自飞书可见正文）

1. **硬件未定论**：仿人 DoF、创新非仿人形态、科研友好平台（如 G1）并存。
2. **数据 scale 可能**：BFM 带来扩数据想象，但质量/采集/筛选矛盾突出（人工 vs 特权学习）。
3. **算法范式待演化**：Physical AI 可能需要不同于 LLM 的范式（监督 vs 无监督 BFM 等）。
4. **Benchmark 缺失**：模仿数据好坏、机器人「有用」指标尚无统一标准。
5. **时间尺度**：约三年从 MPC 主导到 BFM 话题升温（作者观察，非严格史学）。

## 工程实践

- 立项时同时写清 **硬件假设、数据来源、评测协议**，勿只追模型名。
- 跟踪 [BFM 纵深](../../roadmap/depth-bfm.md) 与 [传统纵深](../../roadmap/depth-classical-control.md) 双线能力，避免团队技能单腿走路。

## 局限与风险

- **飞书正文为作者观点 + 参考文献插图**，非系统综述；需与 arXiv/会议论文交叉验证。
- **趋势快变**：2026 中的判断需按季度回看 [log.md](../../log.md) 与新技术地图。

## 关联页面

- [Know-How 技术地图](./humanoid-motion-control-know-how-technology-map.md)
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [运动小脑 64 篇地图](./humanoid-motion-cerebellum-technology-map.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)
- [feishu 抓取摘录](../../sources/raw/feishu_humanoid_motion_control_know_how_2026-07-14.md)

## 推荐继续阅读

- [飞书原文](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87)
