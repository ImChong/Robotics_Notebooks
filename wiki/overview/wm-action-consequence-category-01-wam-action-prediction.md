---
type: overview
tags: [world-models, world-action-models, category-hub, survey]
status: complete
updated: 2026-11-13
summary: "世界模型动作后果专题 · 01 — WAM 直接执行、修正基础 VLA 还是部署前筛选？并链 DreamWAM / FACT / Flex-π / RTCF 邻近坐标。"
related:
  - ./robot-world-models-action-consequence-technology-map.md
  - ./wm-action-consequence-category-02-contact-modeling.md
  - ../concepts/world-action-models.md
  - ../entities/paper-dr-wm.md
  - ../entities/paper-dreamwam.md
  - ../entities/paper-fact.md
  - ../entities/paper-flex-pi.md
  - ../entities/paper-rtcf.md
  - ../entities/paper-motubrain.md
  - ../entities/paper-rift-wam.md
  - ../entities/paper-wam-realtime-async.md
  - ../entities/paper-lawa.md
original:
  - ../../sources/papers/dswam_arxiv_2607_04927.md
  - ../../sources/papers/dynawm_arxiv_2607_02604.md
  - ../../sources/papers/dreamsteer_arxiv_2607_02865.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
  - ../../sources/papers/fact_arxiv_2608_10232.md
  - ../../sources/papers/flex_pi_arxiv_2608_10860.md
  - ../../sources/papers/motubrain_arxiv_2604_27792.md
  - ../../sources/papers/rift_wam_arxiv_2608_11521.md
  - ../../sources/papers/lawa_arxiv_2608_24882.md
  - ../../sources/papers/wam_realtime_async_arxiv_2608_01880.md
  - ../../sources/papers/dr_wm_arxiv_2610_17630.md
---

# 世界模型动作后果分类 01：WAM 动作后果预测

> **分类页**：对应 [具身智能研究室 · 世界模型动作后果专题](https://mp.weixin.qq.com/s/a5ZDDv70CLDfY98mfviWuA) 的 **01 WAM 动作后果预测** 段；总地图见 [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)。

## 一句话定义

按世界模型在控制环中的职责，区分直接生成动作、重生成基础策略动作和部署前筛选候选动作，避免把所有「带未来预测」的方法当作同一种接口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来观测与动作生成的具身策略 |
| VLA | Vision-Language-Action | 常被修正、筛选或后训练的上层策略 |
| WM | World Model | 预测动作后果的潜变量或视频模型 |
| MoE | Mixture of Experts | 异构动作模态的专家混合架构 |

## 为什么重要

同为世界模型增强策略，三类方法对基础 VLA 权重、训练数据和在线候选数量的要求不同。先定接口，再比成功率，才能估计迁移成本与延迟。

## 核心原理：执行、修正与筛选

| 职责 | 训练与部署接口 | 一手证据 |
|---|---|---|
| 直接执行 | 视频/动作共训；部署观测→动作块，规划器可选 | DSWAM §3，折叠对照关闭 System 2 |
| 在线重生成 | 各 base VLA 分别缓存动作并训练 DynaWM；部署 base chunk + 历史→新 chunk | DynaWM §3.5；不共享一套跨 VLA 修正器权重 |
| 部署筛选 | 组件冻结，VLA 样本+原语→WM→VLAC 排序 | DreamSteer §3–4；13 候选串行约 13 秒 |

邻近工作的表格用于导航，具体版本、指标与开放资源以各独立论文节点为准。

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| DR-WM | [DR-WM](../entities/paper-dr-wm.md) | 行为校准的空间地形物理 + Genesis / 视频 rollout，辅助 VLM 比较导航动作后果 |
| DSWAM | [../entities/paper-dswam-dual-system-wam](../entities/paper-dswam-dual-system-wam.md) | 双系统 WAM 执行器 + 可选 VLM 规划器；视频协同训练、推理直出动作 |
| DynaWM | [../entities/paper-dynawm-vla-online-correction](../entities/paper-dynawm-vla-online-correction.md) | 冻结 VLA + Mamba/V-JEPA 条件流匹配在线重生成动作块 |
| DreamSteer | [../entities/paper-dreamsteer-vla-deployment-steering](../entities/paper-dreamsteer-vla-deployment-steering.md) | 潜变量 WM 预演候选动作 + 语言价值模型排序 |
| Worldscape-MoE | [../entities/paper-worldscape-moe-heterogeneous-action](../entities/paper-worldscape-moe-heterogeneous-action.md) | 共享/专属 MoE 统一相机、关节、手部异构动作接口 |
| WorldScape Policy 2.0 | [../entities/paper-worldscape-policy-2](../entities/paper-worldscape-policy-2.md) | 同团队下游 WAM 策略：事件记忆走 VLM、视觉记忆走 DiT，多模态提示可控执行 |
| DreamWAM（邻近） | [../entities/paper-dreamwam](../entities/paper-dreamwam.md) | Joint WAM：beyond-RGB 未来表征训练、RGB-only 部署（非本专题原文四篇） |
| FACT（邻近） | [../entities/paper-fact](../entities/paper-fact.md) | 失败感知因果 WAM：失败轨迹教后果；可选 value 筛选候选 |
| Flex-π（邻近） | [../entities/paper-flex-pi](../entities/paper-flex-pi.md) | 多流 Joint WAM + 算力柔性（56 组合；action-only ~60 ms↔full joint ~193 ms） |
| RTCF（邻近） | [../entities/paper-rtcf](../entities/paper-rtcf.md) | **免训练** 冻结 VLA 记忆纠偏；与 DynaWM「可训修正」对照 |
| Motubrain（邻近） | [../entities/paper-motubrain](../entities/paper-motubrain.md) | 生数 Joint WAM；RoboTwin 95.8/96.1（仓占位） |
| WAM 异步部署（邻近） | [../entities/paper-wam-realtime-async](../entities/paper-wam-realtime-async.md) | 同平台六策略：对齐 → blend → train |
| Rift（邻近） | [../entities/paper-rift-wam](../entities/paper-rift-wam.md) | 免视频 rollout：一次 anticipation prefill 写未来 K/V；LIBERO 98.8% / 247.9 ms |
| LAWA（邻近） | [../entities/paper-lawa](../entities/paper-lawa.md) | 潜动作作测试时未来意图；RoboCasa 65.6/80.8%；相对 Joint −42.9% 延迟 |

## 工程实践

1. 先写当前观测、动作/状态单位与采样率；分清输出是候选动作、未来观测还是价值分数。
2. 分列训练成本与部署关键路径。DSWAM 的视频监督留在训练；DynaWM 每个基础策略需适配权重；DreamSteer 需支付候选数乘 rollout/打分成本。
3. 同协议记录任务成功率、动作忠实度及端到端延迟；模型内单模块时间不能直接当控制频率。对接触/几何，还要区分传感输入、训练 teacher 和推理输出。

## 局限与风险

这三类是职责坐标而非互斥架构。DynaWM 的原文主要是 Isaac Sim 动态拦截，DSWAM 是真机折叠与双臂仿真，DreamSteer 是单臂 OOD 筛选；成功率不可跨协议排名。候选覆盖、视觉误排与观测到执行延迟分别构成不同瓶颈。

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)
- [DreamWAM](../entities/paper-dreamwam.md)
- [FACT](../entities/paper-fact.md)
- [Flex-π](../entities/paper-flex-pi.md)
- [RTCF](../entities/paper-rtcf.md)
- [Motubrain](../entities/paper-motubrain.md)
- [WAM 实时异步部署](../entities/paper-wam-realtime-async.md)
- [Rift](../entities/paper-rift-wam.md)
- [LAWA](../entities/paper-lawa.md)

## 参考来源

- [一手全文核对：dreamsteer_arxiv_2607_02865](../../sources/papers/dreamsteer_arxiv_2607_02865.md)

- [一手全文核对：dynawm_arxiv_2607_02604](../../sources/papers/dynawm_arxiv_2607_02604.md)

- [一手全文核对：dswam_arxiv_2607_04927](../../sources/papers/dswam_arxiv_2607_04927.md)

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)

## 推荐继续阅读

- [DSWAM 原文](https://arxiv.org/abs/2607.04927)
- [DynaWM 原文](https://arxiv.org/abs/2607.02604)
- [DreamSteer 原文](https://arxiv.org/abs/2607.02865)
