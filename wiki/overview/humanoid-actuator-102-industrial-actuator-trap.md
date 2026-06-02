---
type: overview
tags: [humanoid, actuator, industrial-robot, category-hub]
status: complete
updated: 2026-06-02
summary: "Actuator 102 · 06 — 工业执行器优化精度与重复性；人形优化柔顺与生存；平方律使「买货架伺服」在腿上失效。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-gear-reflected-inertia.md
  - ./humanoid-actuator-102-decision-species.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 06：工业执行器陷阱

> **图谱分类节点**：**IX 传统工业执行器为何失效**。

## 核心结论

表面相似（关节+电机+驱动），运行在不同 **物理体系**：

| 工业臂 | 人形腿 |
|--------|--------|
| 准静态、受控环境 | 动态冲击、不可预测接触 |
| 精度、重复性 | 退让、生存、CoT |
| 高减速、自锁丝杠可接受 | 需反向驱动与冲击额定 |

文内观点：**无法直接购买「现成人形腿执行器」**，因货架为另一问题优化。

## 与 N² / 平方律

第九章与第四章衔接：工业 **高减速 + 高反射惯量** 在行走中触发 **平方律级** 失效（冲击×惯量×热）。

## 关联页面

- [决策与物种](./humanoid-actuator-102-decision-species.md)
- [负载与质量螺旋](./humanoid-actuator-102-load-and-mass-spiral.md)

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
