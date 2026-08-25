---
type: overview
tags: [world-models, category-hub, survey, embodied-wm-six-routes]
status: complete
updated: 2026-08-25
summary: "具身世界模型六路线 · 学习主导型 — 训练时在想象经验中优化另一套策略。"
related:
  - ./embodied-wm-six-routes-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
---

# 具身世界模型六路线 · 学习主导型

> **图谱分类节点**：对应 [六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **学习主导型** 段；总地图见 [embodied-wm-six-routes-technology-map](./embodied-wm-six-routes-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测未来状态/观测的内部模型 |
| WAM | World Action Model | 联合未来与动作的具身策略 |
| MPC | Model Predictive Control | 模型内滚动搜索动作 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| World Models (Ha & Schmidhuber) | [paper-ha-schmidhuber-world-models](../entities/paper-ha-schmidhuber-world-models.md) | VAE+MDN-RNN 梦境中训练小控制器。 |
| Dreamer | [paper-dreamer-latent-imagination](../entities/paper-dreamer-latent-imagination.md) | 潜在想象轨迹上训练 actor-critic。 |
| DreamerV3 | [paper-shenlan-wm-13-dreamerv3](../entities/paper-shenlan-wm-13-dreamerv3.md) | 单套算法配置覆盖更丰富任务族的在线想象 RL。 |
| RISE | [paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio](../entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md) | 独立 VLA 在多视角想象环境中生成轨迹并持续更新。 |
| World4RL | [paper-sa-2509-19080-world4rl-diffusion-world-models-for-policy-refin](../entities/paper-sa-2509-19080-world4rl-diffusion-world-models-for-policy-refin.md) | 扩散式操作策略的世界模型后训练。 |
| Robotic World Model | [robotic-world-model-eth-rsl](../entities/robotic-world-model-eth-rsl.md) | 腿足与人形控制上的机器人世界模型想象学习。 |
| DreamGen | [paper-notebook-dreamgen-unlocking-generalization-in-robot-learn](../entities/paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md) | 生成新任务场景机器人视频并补充动作信息训练下游策略。 |
| DayDreamer | [paper-daydreamer-world-models-real-robots](../entities/paper-daydreamer-world-models-real-robots.md) | 世界模型直接在真实机器人上训练策略，无需仿真。 |
| UniSim | [paper-unisim](../entities/paper-unisim.md) | 生成式世界模型作可交互神经环境训练策略。 |
| GigaWorld-0 | [gigaworld-0](../entities/gigaworld-0.md) | 视频外观/动作建模连接 3DGS 与规划，服务 VLA 数据生成。 |
| GR00T-Dreams | [paper-gr00t-dreams-synthetic-trajectories](../entities/paper-gr00t-dreams-synthetic-trajectories.md) | 世界模型扩展机器人轨迹作合成数据。 |

## 关联页面

- [六路线技术地图](./embodied-wm-six-routes-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
