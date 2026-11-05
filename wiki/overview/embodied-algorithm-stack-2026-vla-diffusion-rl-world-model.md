---
type: overview
tags: [overview, vla, diffusion-policy, reinforcement-learning, world-model, algorithm-stack, wechat]
status: complete
updated: 2026-11-04
related:
  - ../entities/paper-openvla.md
  - ../entities/paper-octo.md
  - ../entities/paper-smolvla.md
  - ../entities/paper-pi0.md
  - ../entities/paper-pi05-open-world-vla.md
  - ../entities/paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md
  - ../methods/vla.md
  - ../methods/diffusion-policy.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md
summary: "新机器视觉转载综述的阅读坐标：2026 具身智能四层算法栈（VLA / 扩散策略 / RL / 世界模型），文中点名的算法逐个回到一手资料核对（15 复用 / 3 新建 / 2 无法定位），并指出文章与 README、论文不一致之处。"
---

# 2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）

> **本页定位**：为 [2026年了，主流具身智能算法有哪些？](https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg)（新机器视觉转载，2026-11-04）提供阅读坐标。文章是没有出处的转载综述，本页对每个算法回到一手资料核对，**以一手资料为准、文章说法仅作对照**。

## 一句话观点

**文章的结论：别追求单一算法通吃——VLA 当基座、扩散策略保可靠性、RL 啃硬技能、世界模型赌未来泛化，按场景的可靠性、算力与数据条件组合。**（文章观点，工程上是合理的分层叙事；其中具体数字有若干与一手资料不一致，见下。）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| NFE | Number of Function Evaluations | 扩散采样的神经函数评估次数 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化（WMPO 使用） |
| RECAP | RL with Experience and Corrections via Advantage-conditioned Policies | π*0.6 的训练方法 |

## 为什么单独做这张地图

- 库内相关节点散落各处，且存在**同名不同篇**的陷阱（如 S²-VLA 在库内已有驾驶版、FlowVLA 已有两篇别的论文），需要一张明确对应关系的表。
- 文章的若干具体数字与一手资料不符（Octo 推理频率、OpenVLA 适配硬件与许可），必须显式标注。

## 四层算法栈与节点

| 文中名称 | 层 | 节点 |
|----------|----|------|
| OpenVLA | VLA | [paper-openvla](../entities/paper-openvla.md)（复用） |
| Octo | VLA | [paper-octo](../entities/paper-octo.md)（复用） |
| SmolVLA | VLA | [paper-smolvla](../entities/paper-smolvla.md)（复用） |
| π0 | VLA | [paper-pi0](../entities/paper-pi0.md)（复用） |
| π0.5 | VLA | [paper-pi05-open-world-vla](../entities/paper-pi05-open-world-vla.md)（复用） |
| π0.6（π*0.6） | VLA | [paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience](../entities/paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md)（复用） |
| π0.7 | VLA | [pi07-policy](../methods/pi07-policy.md)（复用） |
| HiFOR | VLA | [paper-hifor-hierarchical-foresight-vla](../entities/paper-hifor-hierarchical-foresight-vla.md)（复用） |
| S²-VLA | VLA | [paper-s2-vla-state-space-guided](../entities/paper-s2-vla-state-space-guided.md)（新建） |
| FlowVLA | VLA | [paper-flowvla-visual-cot](../entities/paper-flowvla-visual-cot.md)（新建） |
| Diffusion Policy | 扩散策略 / 模仿学习 | [paper-diffusion-policy](../entities/paper-diffusion-policy.md)（复用） |
| ACT | 扩散策略 / 模仿学习 | [paper-act](../entities/paper-act.md)（复用） |
| Two-Steps Diffusion Policy | 扩散策略 / 模仿学习 | [paper-two-steps-diffusion-policy](../entities/paper-two-steps-diffusion-policy.md)（新建） |
| REFINE-DP | 扩散策略 / 模仿学习 | [paper-loco-manip-161-157-refine-dp](../entities/paper-loco-manip-161-157-refine-dp.md)（复用） |
| PPO | 强化学习 | [ppo](../methods/ppo.md)（复用） |
| SuperMimic | 强化学习 | [paper-supermimic](../entities/paper-supermimic.md)（复用） |
| ELEVATE | 强化学习 | [paper-elevate](../entities/paper-elevate.md)（复用） |
| WMPO | 世界模型 | [paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v](../entities/paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v.md)（复用） |

## 无法定位一手论文的两项

| 名称 | 核对结果 |
|------|----------|
| CSO | 文章未给全称与出处；我用关键词搜索没有找到与之对应的机器人论文（命中的 CSO 是 LLM 智能体方法，Skill1 的 97.5% 是 ALFWorld，均不是），**未能定位一手论文，不建节点** |
| Latent PPO | 文章未给出处；搜索没有找到名为 Latent PPO 的论文（最接近的 ATRos 为 PPO + 潜在嵌入辅助，是否即所指无法确认），**不建节点** |

## 场景选型建议（文章）

| 场景 | 推荐组合 | 理由 |
|------|----------|------|
| 工业分拣 / 上下料 | 轻量 VLA（Octo / SmolVLA）+ Diffusion Policy + 少量 RL 精修 | 边界清晰、结果可量化，优先速度与稳定性 |
| 双臂遥操作 / 柔性装配 | ACT + VLA 基座微调 | 训练快、演示效率高，适合快速迭代 |
| 腿足巡检 / 搬运 | PPO / Latent PPO + 世界模型预训练 | 运动控制靠 RL，长程规划靠世界模型降本 |
| 家庭服务长程任务 | π0 系列 / OpenVLA + 世界模型 + 具身记忆 | 需要分层规划、高频控制与跨任务泛化 |
| 灵巧手精细操作 | Flow Matching 动作头 + 扩散策略 + RL 失败驱动 | 高频控制 + 多模态动作分布 + 长尾技能精修 |

## 文章的市场数据（作者表述，未核实）

- 智元「精灵 G2」在龙旗科技工厂 6 天测试累计 64,828 次作业、综合成功率 99.99%
- 2026 年上半年智元出货约 9,700 台、占全球人形机器人市场超过 43%
- 临界点灵巧手在真实工位完成 72 道工序、成功率 98%、年量产突破 15,000 台、整机故障率低于 0.5%
- 2026 年上半年国内具身智能融资额 934 亿元、为 2025 年同期的五倍；中国具身智能市场规模预计 2026 年达 1.09 万亿元
- 全球人形机器人出货结构中娱乐表演、科研仍占 60% 以上，智能制造和仓储物流分别仅 13% 和 5%

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [强化学习](../methods/reinforcement-learning.md)
- [VLA 演进简史（Khushi 长文）](./vla-history-rt1-to-molmoact-technology-map.md)

## 参考来源

- [wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md](../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md)
