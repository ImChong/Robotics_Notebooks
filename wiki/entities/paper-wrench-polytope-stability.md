---
type: entity
tags:
  - paper
  - locomotion
  - contact
  - stability
  - mpc
status: complete
updated: 2026-09-20
arxiv: "2609.17405"
related:
  - ../concepts/contact-dynamics.md
  - ../tasks/locomotion.md
  - ../methods/model-predictive-control.md
sources:
  - ../../sources/papers/wrench-polytope-stability_arxiv_2609_17405.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md
summary: "力旋多面体稳定（arXiv:2609.17405）：高效六维多面体交集与原点单纯形扩展；~49 Hz 计算各关节可实现力矩；LAURON VI 斜墙多接触稳定。"
---

# 力旋多面体稳定（arXiv:2609.17405）

**力旋多面体稳定**（*Optimized Wrench Polytope Analysis for Real-Time Stability Control of Legged Robots in Complex Multi-Contact Configurations*，[arXiv:2609.17405](https://arxiv.org/abs/2609.17405)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)（2026-09-14–18）。

## 一句话定义

**高效六维多面体交集与原点单纯形扩展；~49 Hz 计算各关节可实现力矩；LAURON VI 斜墙多接触稳定。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Wrench | Wrench Polytope | 力旋多面体 |
| MPC | Model Predictive Control | 模型预测控制 |
| CoM | Center of Mass | 质心 |

## 为什么重要

- 复杂多接触下实时力旋分析是稳定控制瓶颈。

## 流程总览

```mermaid
flowchart TD
    A["各足接触 c：摩擦模型或实测滑移阈值"] --> B["摩擦多面体 FP_c (V 表示)"]
    C["关节力矩限位 + RNEA 逆动力学变换 T_ID"] --> D["可驱动多面体 AP_c (H 表示)<br/>加力矩为 0 的虚拟关节滤除奇异方向"]
    B --> E{"接触构型改变？"}
    D --> E
    E -->|是| F["精确求交 FP_c ∩ AP_c<br/>(H 表示求交后转回 V 表示)"]
    E -->|否| G["缩放法：超限顶点按比例缩放到 AP_c 边界"]
    F --> H["接触 wrench 多面体 CW_c"]
    G --> H
    H --> I["叠加 CoM 力矩关系 r_c × f，得到 CWm_c"]
    I --> J["不展开完整 Minkowski 和<br/>目标 wrench w_goal 作为输入"]
    J --> K["找含原点的初始 simplex"]
    K --> L["沿与 w_goal 相交的面法向<br/>搜极值点并替换 simplex 顶点"]
    L --> M{"出口面已在 FW 边界？"}
    M -->|否| L
    M -->|是| N["把交点缩放到 w_goal 长度，插值得到各足 wrench"]
    N --> O["T_ID 转为各关节力矩"]
    P["位姿误差 → PID → 基座目标 wrench"] --> J
```

图依据：论文 §III–V（Fig. 2–5）与 §VI 控制器架构（Fig. 6）

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.17405](https://arxiv.org/abs/2609.17405) |
| **开源** | **待发布**（步骤 2.5，2026-09-20） |
| **方法摘要** | Optimized wrench polytope intersection + feasible torque computation at ~49 Hz. |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。

## 实验与评测

- LAURON VI multi-contact incl. inclined wall support（作者报告）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**本文把力旋多面体分析推到复杂多接触场景的实时稳定控制。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [contact-dynamics](../concepts/contact-dynamics.md)
- [locomotion](../tasks/locomotion.md)
- [model-predictive-control](../methods/model-predictive-control.md)

## 参考来源

- [wrench-polytope-stability_arxiv_2609_17405.md](../../sources/papers/wrench-polytope-stability_arxiv_2609_17405.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)
- [arXiv:2609.17405](https://arxiv.org/abs/2609.17405)
- [arXiv PDF](https://arxiv.org/pdf/2609.17405)：已通读方法部分（实验与附录以略读为准）：§I–V（稳定判据、CW_c 构造、FW 迭代分析），§VI 控制器架构与实验设定

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.17405)
