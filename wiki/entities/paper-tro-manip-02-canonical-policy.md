---
type: entity
tags: [paper, manipulation, tro-manip-survey, se3-equivariant, point-cloud, zju, purdue]
status: complete
updated: 2026-07-08
arxiv: "2505.18474"
summary: "点云观测映射到规范化 3D 坐标系，使策略学习满足 SE(3) 等变；仿真 +18.0%、真机 +39.7%。"
related:
  - ../overview/tro-manip-5-papers-technology-map.md
  - ../overview/tro-manip-category-02-representation.md
  - ../methods/diffusion-policy.md
  - ../methods/behavior-cloning.md
sources:
  - ../../sources/papers/tro_manip_survey_02_canonical_policy.md
  - ../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md
  - ../../sources/papers/tro_manip_5_papers_catalog.md
---

# Canonical Policy

**Canonical Policy** 收录于 [深蓝具身智能 · T-RO 2026 操作学习精选](https://mp.weixin.qq.com/s/nswA-jCGC3kr9iQjhRRuXQ) **第 02/5** 篇，归类为 **02 三维与手物表征**。

## 一句话定义

通过 **规范化 3D 表征** 将点云观测统一映射到标准参考坐标系，使观测→动作映射 **SE(3) 等变**，在未见物体姿态与视角下仍保持泛化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SE(3) | Special Euclidean Group in 3D | 三维刚体旋转+平移变换群 |
| IL | Imitation Learning | 模仿学习 |
| OOD | Out-of-Distribution | 分布外泛化 |

## 为什么重要

- 传统模仿学习需海量数据覆盖物体位姿变化；Canonical Policy 从 **几何对称性** 而非记忆特定姿态提取泛化。
- 建立 **三维规范化表征理论**，比非结构化堆叠等变组件更可解释。
- 可接现代 **生成式策略**（如扩散模型）作为策略头，兼顾对称性与表达力。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 02/5 |
| 分组 | 02 三维与手物表征 |
| 机构 | 浙江大学（ZJU）；普渡大学（Purdue） |
| 出处 | IEEE T-RO 2026 · arXiv:2505.18474 |
| 论文/项目 | <https://arxiv.org/abs/2505.18474> · <https://zhangzhiyuanzhang.github.io/cp-website/> |

## 核心机制（归纳）

### 1）规范化映射

- 将点云观测 **归一化** 到标准坐标系下决策，再将动作 **反变换** 回真实空间执行。
- 无论物体如何旋转/平移，策略在 canonical 空间内学习 **与姿态无关** 的映射。

### 2）实验亮点（策展）

- 12 个仿真任务 + 4 个真机任务，16 种配置（颜色/形状/视角/平台）。
- 相对 SOTA 模仿学习策略：仿真 **+18.0%**，真机 **+39.7%**（文内/report 归纳）。

## 常见误区

1. SE(3) 等变 ≠ 无需任何演示数据——仍须足够 canonical 空间内的示教覆盖任务语义。
2. 与 EquiBot / DP3 等 3D 策略对照时，须区分 **canonicalization** 与 **网络内置等变层** 的不同实现路径。

## 实验与评测

- 仿真与真机多配置验证；完整消融与 baseline 以项目页 / PDF 为准。

## 与其他页面的关系

- 技术地图：[tro-manip-5-papers-technology-map.md](../overview/tro-manip-5-papers-technology-map.md)
- 分类 hub：[tro-manip-category-02-representation.md](../overview/tro-manip-category-02-representation.md)
- [Diffusion Policy](../methods/diffusion-policy.md)

## 参考来源

- [tro_manip_survey_02_canonical_policy.md](../../sources/papers/tro_manip_survey_02_canonical_policy.md)
- [wechat_shenlan_tro_manip_5_papers_survey.md](../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
- 论文：<https://arxiv.org/abs/2505.18474>

## 推荐继续阅读

- [DexRepNet++（同组姊妹篇）](./paper-tro-manip-03-dexrepnet-plus-plus.md)
- [T-RO 5 篇技术地图](../overview/tro-manip-5-papers-technology-map.md)
