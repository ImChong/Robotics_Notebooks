---
type: overview
tags: [humanoid, actuator, reflected-inertia, qdd, backdrivability, category-hub]
status: complete
updated: 2026-11-21
summary: "Actuator 102 · 03 — N² 反射惯量：100:1 减速使输出端惯量放大 10000 倍；透明性光谱从谐波到 QDD，决定绊障退让与 RL 力感知。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-split-architecture.md
  - ./humanoid-hardware-101-actuation-sensing-chain.md
  - ./humanoid-actuator-102-compliance-sensing.md
  - ./humanoid-actuator-102-industrial-actuator-trap.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
original:
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 03：减速与反射惯量

> **分类页**：**IV 齿轮减速权衡**。

## 一句话定义

减速与反射惯量权衡是指减速比 $N$ 虽然以线性比例（$\propto N$）放大电机输出力矩，但同时会把电机转子自身的转动惯量 $J_m$ 按减速比平方（$\propto N^2$）反射到关节输出端，从而直接决定关节在绊障时能否快速退让、冲击是否摧毁齿面以及扭矩传感的透明性光谱。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct Drive | 准直驱，依托低减速比（约 6:1–12:1）压低 $N^2$ 惯量的作动架构 |
| RL | Reinforcement Learning | 强化学习，高透明低惯量关节可提高触地反馈信号保真度 |
| RV | Rotate Vector Reducer | 精密摆线针轮行星复合减速器，常见于工业重臂 |

## 为什么重要

许多关节设计者在选型时只做静态转矩除法（所需力矩 / $N$ = 电机标称力矩），往往为了用一颗微小高转速电机而轻易把减速比拉高到 100:1 以上。但在动态双足奔跑或突遇台阶绊脚时，由于 $N^2$ 的平方级惯量惩罚，原本轻巧的转子在关节输出端呈现为一堵巨大的“机械飞轮墙”，在电机还来得及转动之前就已经把冲击动量砸在齿轮啮合点上。

## 核心原理与光谱分布

### $N^2$ 陷阱

设减速比为 $N$，电机转子惯量为 $J_m$：输出力矩 $\tau_{\text{out}} \propto N$，而等效到关节输出端的转子反射惯量满足：
$$J_{\text{reflected}} = N^2 J_m$$

- 以常见的 **100:1** 减速器为例：出力虽然放大 100 倍，但在外力反推视野里电机转子惯量被瞬间放大 **10000×**。
- 当机器人摆动腿绊到台阶或落地踩空时，腿需要**亚毫秒级顺应退让**；若反射惯量过高，整条腿像一根刚性钢杆，冲击峰值应力直接在齿面剪切爆发。

### 透明性光谱（Transparency Spectrum）

- **低透明 / 高减速端**：传统 **100:1+ 谐波 / RV 减速器**，具备极高的静态紧凑扭矩密度与零背隙优势，但反向驱动摩擦与高 $N^2$ 惯量使其几乎丧失无传感器透明感知能力。
- **高透明 / 低减速端**：**QDD（典型 6:1–12:1 行星）**，反射惯量压缩百倍以上，电机绕组电流与外力矩高度强线性关联，最有利于高动态足式碰撞吸收与强化学习（RL）接触感知。
- 本节从**动态行走退让**视角呼应 [Hardware 101 · 传动链](./humanoid-hardware-101-actuation-sensing-chain.md) 对 QDD 与谐波的机械特性对比。

## 工程实践

- **合理控制减速比与转子直径权衡**：人形动作用旋转减速比常取 **6:1–50:1**（按关节承担自由度分化）；在 `70 kg` 人形参考尺度下，输出端等效反射惯量通常应控制在 **< 0.1 kg·m²**。
- **大转矩电机配适中减速优于高速小电机配超大减速**：原文强调的经典设计铁律——宁可用转矩常数更大、外径稍大但转子转动惯量优化好的低极槽空心转子匹配中等减速（如 10:1–30:1），也不要试图用极小高速航模/伺服电机叠加 100:1 以上高比箱来强凑扭矩。
- **台架反推标定等效惯量**：装机测试阶段不能只看 CAD 转子惯量数值，还需包含联轴器、波形发生器与太阳轮轴颈，并在空载扫频台架上实测验证输出端阻尼与反向驱动阈值。

## 常见误区或局限

- **误区：只要电机转子本身很小很轻，用 120:1 减速照样没多大惯量** — 在平方函数面前，一个微小的 $J_m$ 乘以 $14400$ 后往往比整条小腿连杆自身的绕关节转动惯量还要大得多。
- **误区：反射惯量越小就越应该全局强上 QDD** — 在肩臂末端持重或大负载准静态支撑场合，如果强制把减速比压得极低，会导致电机本体为了扛稳重力矩而严重增重并产生极大 $I^2 R$ 铜损发热，反过来诱发质量与热学失控。

## 关联页面

- [Actuator 102 技术地图](./humanoid-actuator-102-technology-map.md)
- [旋转-直线分离架构](./humanoid-actuator-102-split-architecture.md)
- [Hardware 101 · 传动链](./humanoid-hardware-101-actuation-sensing-chain.md)
- [柔顺与感知反馈](./humanoid-actuator-102-compliance-sensing.md)
- [工业执行器陷阱](./humanoid-actuator-102-industrial-actuator-trap.md)

## 推荐继续阅读

- [Actuator 102 · 04：热学与力矩控制](./humanoid-actuator-102-thermal-and-control.md) — 与低减速相伴而来的铜损热挑战
- [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md) — 评估低电感、低惯量高槽满率无框电机

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)（第四章）
