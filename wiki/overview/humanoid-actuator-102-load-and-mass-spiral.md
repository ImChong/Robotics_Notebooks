---
type: overview
tags: [humanoid, actuator, cost-of-transport, fatigue, category-hub]
status: complete
updated: 2026-06-02
summary: "Actuator 102 · 01 — 每小时约 5000 步、2–3× 体重冲击、亚毫秒退让；CoT 与比力矩门槛（>10–15 Nm/kg）。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-split-architecture.md
  - ./humanoid-hardware-101-integrated-actuators.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 01：负载与质量螺旋

> **图谱分类节点**：**I 行走难题** + **II 质量惩罚螺旋**；总地图见 [Humanoid 执行器 102](./humanoid-actuator-102-technology-map.md)。

## 核心问题

为何工业额定执行器在人形腿上数周失效？

| 机制 | 文内量级 |
|------|----------|
| **步频×时长** | ~84 步/分 → **~5000 冲击/小时**；8 小时班 **>4 万次** |
| **冲击幅值** | 每步 **2–3× 体重**；heel strike **50–100 ms** 内 1400–2100 N（70 kg） |
| **响应窗口** | 快于传感器环 → 需 **机械反向驱动** 吸收，否则减速器剪切失效 |
| **CoT** | 双足 **0.2–0.5** vs 轮式 **0.01–0.05**；每克质量放大抬腿能耗 |

## 静态 vs 动态额定

- 工业丝杠按 **准静态** 额定；行走是 **反复接住下落重物**。
- 滚珠丝杠点接触 → **布氏压痕** 累积失效（文内对比滚柱线接触）。

## 比力矩 / 比力门槛

- 旋转主关节：**比力矩 >10 Nm/kg**（竞争力 >25）
- 直线次要：**比力 >4000 N/kg**

## 质量惩罚螺旋（要点）

超重执行器 → 更大电机/电池 → 更高 CoT → 更重结构 → **指数级** 恶化；200 g 超额可经杠杆放大为髋膝 **数 kg 等效负载**。

## 关联页面

- [分离架构](./humanoid-actuator-102-split-architecture.md)
- [Humanoid Hardware 101 · 集成执行器](./humanoid-hardware-101-integrated-actuators.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CoT | Cost of Transport | 单位重量·距离能耗的无量纲移动效率指标 |

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
