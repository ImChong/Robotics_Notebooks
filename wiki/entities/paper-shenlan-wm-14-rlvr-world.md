---
type: entity
tags:
- paper
- world-models
- shenlan-survey
- open-source
- tsinghua
- awesome-world-action-models-rcl
- rcl-wam-catalog
status: complete
updated: 2026-10-06
arxiv: '2505.13934'
venue: NeurIPS 2025
summary: RLVR 优化 WM 对齐任务成功指标，让想象更有目的性。
related:
- ../overview/world-models-15-open-source-technology-map.md
- ../overview/world-models-route-03-virtual-sandbox.md
- ../overview/robot-world-models-training-loop-taxonomy.md
- ../methods/generative-world-models.md
- ../concepts/world-action-models.md
- ./paper-dash-opsd.md
- paper-rcl-wam-robot-learning-control-survey.md
- ../overview/rcl-awesome-wam-technology-map.md
- ../methods/vla.md
- ../tasks/manipulation.md
- ../tasks/locomotion.md
sources:
- ../../sources/papers/shenlan_wm_survey_14_rlvr-world.md
- ../../sources/papers/shenlan_world_models_15_reference_catalog.md
- ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
- ../../sources/papers/rcl_awesome_wam_ref_a8d6b1d31a7e72a424da_rlvr-world-training-world-models-with-re.md
- ../../sources/papers/rcl_awesome_wam_catalog.md
- ../../sources/repos/awesome-world-action-models-rcl.md
project_id: shenlan-wm-14-rlvr-world
code: https://github.com/thuml/RLVR-World
---

# RLVR-World

**RLVR-World** 收录于 [深蓝具身智能 · 世界模型 15 开源项目专题](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg) **第 14/15** 篇，归类为 **03 虚拟沙盒**。

## 一句话定义

RLVR 优化 WM 对齐任务成功指标，让想象更有目的性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |

| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |

## 为什么重要

- RLVR 优化 WM 对齐任务成功指标，让想象更有目的性。
- 属于 [世界模型 15 项目地图](../overview/world-models-15-open-source-technology-map.md) **路线 03**。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 14/15 |
| 路线 | 03 虚拟沙盒 |
| 出处 | NeurIPS 2025 |
| 文内引用 | 34（2026-06-02，策展） |
| arXiv | [2505.13934](https://arxiv.org/abs/2505.13934) |

## 核心机制（归纳）

### 1）策展导读要点

世界模型作为 RL/评估虚拟环境，在想象中 rollout 替代昂贵真机试错；强调物理一致性与下游策略增益。

### 2）策展导读要点

RLVR 优化 WM 对齐任务成功指标，让想象更有目的性。

## 结论

**RLVR-World 换掉的是世界模型的优化目标：不再以还原未来画面为终点，而是用 RLVR 把「想象」直接对齐到任务成功指标，让 rollout 变得有目的性。**

- 真正起作用的是目标函数而非骨干：世界模型被当作 RL/评估的虚拟环境，用可验证奖励微调后，想象中的 rollout 才能替代昂贵真机试错。
- 成败判据随之改变——不看生成质量，而看 **物理一致性与下游策略增益**；这也是 [路线 03 虚拟沙盒](../overview/world-models-route-03-virtual-sandbox.md) 共同的评价口径。
- 使用边界：本页为清单索引（NeurIPS 2025），量化指标以原文为准；引用量与控制一致性无简单线性关系，复现前需核对 License 与权重。

## 常见误区

1. 开源 WM 项目的引用量与 **控制一致性/下游任务增益** 无简单线性关系；复现前需核对 License 与权重。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 路线 hub：[world-models-route-03-virtual-sandbox.md](../overview/world-models-route-03-virtual-sandbox.md)
- 总地图：[world-models-15-open-source-technology-map.md](../overview/world-models-15-open-source-technology-map.md)
- 原始 source：[shenlan_wm_survey_14_rlvr-world.md](../../sources/papers/shenlan_wm_survey_14_rlvr-world.md)
- RLVR / 推理后训练对照：[DASH](./paper-dash-opsd.md) — 在 OPSD 上做分歧自适应蒸馏聚合（数学推理 LM，非世界模型）

## 参考来源

- [shenlan_wm_survey_14_rlvr-world.md](../../sources/papers/shenlan_wm_survey_14_rlvr-world.md)
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)
- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)

- [`sources/papers/rcl_awesome_wam_ref_a8d6b1d31a7e72a424da_rlvr-world-training-world-models-with-re.md`](../../sources/papers/rcl_awesome_wam_ref_a8d6b1d31a7e72a424da_rlvr-world-training-world-models-with-re.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://proceedings.neurips.cc/paper_files/paper/2025/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html>

- [原论文与官方资源](https://github.com/thuml/RLVR-World)

## 推荐继续阅读

- [arXiv:2505.13934](https://arxiv.org/abs/2505.13934) — 论文全文
- [深蓝具身智能原文](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg)

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html)

## 源码运行时序图

**不适用**（本次合并的来源仅归档论文策展与官方仓库地址，尚未核验训练/推理入口；不能据清单编造实现时序。复现前须补充 README 运行步骤与源码模块归档）。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

- [robot-world-models-training-loop-taxonomy](../overview/robot-world-models-training-loop-taxonomy.md)
- [world-action-models](../concepts/world-action-models.md)
- [awesome-world-action-models-rcl](paper-rcl-wam-robot-learning-control-survey.md)
- [vla](../methods/vla.md)
- [locomotion](../tasks/locomotion.md)
