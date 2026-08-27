---
type: overview
tags: [world-models, world-action-models, category-hub, survey]
status: complete
updated: 2026-08-27
summary: "世界模型动作后果专题 · 04 — 策略评估与世界模型进入研发链路"
related:
  - ./robot-world-models-action-consequence-technology-map.md
  - ./wm-action-consequence-category-01-wam-action-prediction.md
  - ../concepts/world-action-models.md
  - ../entities/paper-sc3-eval.md
  - ../entities/paper-worldecho-worldsync.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
  - ../entities/current-robotics-currentworld.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
  - ../../sources/papers/sc3_eval_arxiv_2606_18610.md
  - ../../sources/papers/worldecho_worldsync_arxiv_2608_24885.md
  - ../../sources/blogs/current_robotics_currentworld.md
---

# 世界模型动作后果分类 04：训练与评估闭环

> **图谱分类节点**：对应 [具身智能研究室 · 世界模型动作后果专题](https://mp.weixin.qq.com/s/a5ZDDv70CLDfY98mfviWuA) 的 **04 训练与评估闭环** 段；总地图见 [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来观测与动作生成的具身策略 |
| VLA | Vision-Language-Action | 常被修正、筛选或后训练的上层策略 |
| WM | World Model | 预测动作后果的潜变量或视频模型 |
| MoE | Mixture of Experts | 异构动作模态的专家混合架构 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| GigaWorld-1 | [../entities/paper-gigaworld-1-policy-evaluation](../entities/paper-gigaworld-1-policy-evaluation.md) | 7 类视频 WM + 32 万轨迹研究策略评估对齐 |
| SC3-Eval | [../entities/paper-sc3-eval](../entities/paper-sc3-eval.md) | 自一致视频生成作真机 VLA 评估器（闭环 \(r=0.929\)） |
| WorldEcho / WorldSync | [../entities/paper-worldecho-worldsync](../entities/paper-worldecho-worldsync.md) | off-expert 动作跟随评测 + AFE/IE 对齐；作策略改进模拟器（确认未开源） |
| CurrentWorld-0 | [../entities/current-robotics-currentworld](../entities/current-robotics-currentworld.md) | 产业侧交互模拟器：评测 + 失败态回滚分支后训练（确认未开源） |


### 交叉引用（文内第四节）

| 工作 | Wiki 实体 | 角色 |
|------|-----------|------|
| DreamSteer | [../entities/paper-dreamsteer-vla-deployment-steering](../entities/paper-dreamsteer-vla-deployment-steering.md) | 部署前候选动作筛选 |
| TACO | [../entities/paper-taco-tactile-wm-vla-posttrain](../entities/paper-taco-tactile-wm-vla-posttrain.md) | 失败片段纠错后训练 |
| EmbodiedGen V2 | [../entities/paper-embodiedgen-v2-sim-ready-world-engine](../entities/paper-embodiedgen-v2-sim-ready-world-engine.md) | 环境与数据扩展 |

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)
- [CurrentWorld-0](../entities/current-robotics-currentworld.md) — 跨本体交互 WM 作评测/后训练沙盒
- [WorldEcho / WorldSync](../entities/paper-worldecho-worldsync.md) — 动作跟随评测与对齐配方（arXiv:2608.24885）

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [current_robotics_currentworld.md](../../sources/blogs/current_robotics_currentworld.md)
