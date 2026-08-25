---
type: overview
tags: [world-models, category-hub, survey, embodied-wm-six-routes]
status: complete
updated: 2026-08-25
summary: "具身世界模型六路线 · 规划主导型 — 执行时在模型中试走，外部规划/验证裁决动作。"
related:
  - ./embodied-wm-six-routes-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
---

# 具身世界模型六路线 · 规划主导型

> **图谱分类节点**：对应 [六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **规划主导型** 段；总地图见 [embodied-wm-six-routes-technology-map](./embodied-wm-six-routes-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测未来状态/观测的内部模型 |
| WAM | World Action Model | 联合未来与动作的具身策略 |
| MPC | Model Predictive Control | 模型内滚动搜索动作 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| Visual Foresight | [paper-visual-foresight-latent-mpc](../entities/paper-visual-foresight-latent-mpc.md) | 动作条件视频预测接入真实机器人视觉 MPC，奠定预测后果—在线选择—重规划范式。 |
| PETS | [paper-pets-probabilistic-dynamics-mpc](../entities/paper-pets-probabilistic-dynamics-mpc.md) | 概率动力学集成+不确定性传播增强 CEM，真机样本高效 MBRL 代表。 |
| Resilient Machines (Self-Modeling) | [paper-resilient-machines-continuous-self-modeling](../entities/paper-resilient-machines-continuous-self-modeling.md) | 从动作—感觉关系推断自身结构，损伤后搜索替代行为；身体也是世界的一部分。 |
| PlaNet | [paper-planet-latent-dynamics](../entities/paper-planet-latent-dynamics.md) | 潜状态压缩视觉观测，执行期 CEM 在内部搜索动作序列。 |
| MuZero | [paper-muzero-planning-latent-dynamics](../entities/paper-muzero-planning-latent-dynamics.md) | 只学树搜索需要的奖励/策略/价值，不必还原真实画面。 |
| V-JEPA 2 | [paper-vjepa2](../entities/paper-vjepa2.md) | 互联网视频预训练表征，V-JEPA 2-AC 用少量机器人轨迹做动作条件 latent MPC。 |
| TD-MPC2 | [paper-td-mpc2](../entities/paper-td-mpc2.md) | 短时域潜空间规划+终端价值补足远期回报。 |
| DINO-WM | [paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu](../entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md) | 在视觉基础模型特征空间直接预测未来。 |
| RoboCraft | [paper-robocraft-particle-graph-dynamics](../entities/paper-robocraft-particle-graph-dynamics.md) | 粒子图预测弹塑性物体形变与接触。 |
| ParticleFormer | [paper-sa-2506-23126-particleformer-a-3d-point-cloud-world-model-for](../entities/paper-sa-2506-23126-particleformer-a-3d-point-cloud-world-model-for.md) | 多物体多材料交互的三维点云世界模型。 |
| PointWorld | [paper-sa-2601-03782-pointworld](../entities/paper-sa-2601-03782-pointworld.md) | 三维点流统一场景变化与跨本体动作，由 MPC 调用。 |
| τ₀-VLA | [paper-tau0-vla](../entities/paper-tau0-vla.md) | 高层不确定时 VLM 提子任务，WM 预测分支后果并由价值模型 beam search。 |
| CheckVLA | [paper-checkvla-execution-time-verification](../entities/paper-checkvla-execution-time-verification.md) | 执行时用动作条件 WM 比较预测与真实观测，风险越界则改写后续动作。 |

## 关联页面

- [六路线技术地图](./embodied-wm-six-routes-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
