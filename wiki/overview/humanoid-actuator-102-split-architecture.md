---
type: overview
tags: [humanoid, actuator, harmonic-drive, roller-screw, category-hub]
status: complete
updated: 2026-06-02
summary: "Actuator 102 · 02 — 重载通用人形趋同：肩髋旋转用谐波，膝踝冲击用行星滚柱丝杠直线；滚珠丝杠点接触易布氏压痕。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-gear-reflected-inertia.md
  - ./humanoid-hardware-101-linear-transmission-bearings.md
  - ./humanoid-hardware-101-actuation-sensing-chain.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 02：旋转-直线分离架构

> **图谱分类节点**：**III 趋同解法**。

## 核心结论

Tesla、Figure、Apptronik 等 **重载通用人形** 独立趋同：**旋转谐波**（肩腕髋旋转）+ **直线行星滚柱丝杠**（膝踝等冲击关节）。非唯一架构：Unitree 偏 **高动态旋转**；Digit 偏 **SEA**；Atlas 电动变体各异。

## 旋转：谐波齿轮

- 柔轮弹性变形、**零背隙**、单级 **50:1–100:1**、高力矩密度。
- 代价：**效率低于行星**、柔性件疲劳与发热（见 [热学与控制](./humanoid-actuator-102-thermal-and-control.md)）。

## 直线：行星滚柱丝杠

- **线接触** vs 滚珠 **点接触** → 峰值赫兹应力约低一个数量级。
- 行走冲击下滚珠易 **布氏压痕**；滚柱额定循环在冲击工况可差 **10× 以上**（文内数量级表述）。

## 策略性分离

| 执行器 | 擅长 | 不擅长 |
|--------|------|--------|
| 谐波旋转 | 零背隙、紧凑力矩 | 反复冲击疲劳、低效率发热 |
| 滚柱直线 | 冲击、高力密度 | 高效旋转 |

主关节 **旋转执行器** 为主（Optimus、Figure、Unitree、BD 等）；直线用于手指肌腱、头/躯干等次要自由度。

## 关联页面

- [减速与反射惯量](./humanoid-actuator-102-gear-reflected-inertia.md)
- [Hardware 101 · 直线与轴承](./humanoid-hardware-101-linear-transmission-bearings.md)

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
