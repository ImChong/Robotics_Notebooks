---
type: entity
tags:
  - paper
  - world-models
  - deformable
  - differentiable-physics
  - mpm
  - residual-learning
  - georgia-tech
status: complete
updated: 2026-08-10
arxiv: "2607.20653"
related:
  - ../overview/world-model-physics-fidelity-outputs.md
  - ./paper-core.md
  - ./core-retarget.md
  - ../methods/generative-world-models.md
  - ../concepts/kinematic-vs-dynamic-feasibility.md
  - ./paper-vt-wam-visuotactile-contact-rich.md
  - ./paper-masked-visual-actions.md
  - ./paper-imagined-rollouts-kinematic-not-dynamic.md
  - ./paper-kinebench.md
  - ../concepts/physics-fidelity-sim2real-gap.md
sources:
  - ../../sources/papers/physcore_arxiv_2607_20653.md
  - ../../sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md
summary: "PhysCoRe（arXiv:2607.20653，Georgia Tech）：可微 MLS-MPM + Material from Motion（MfM）推断粒子材料与置信度 + Residual from Dynamics（RfD）修正接触/摩擦/模型误差；真机可变形序列相对 PhysTwin 显著降 CD；截至 2026-07-27 未开源。"
---

# PhysCoRe（物理修正残差世界模型 · 材料感知可变形动力学）

**PhysCoRe**（*Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics*，[arXiv:2607.20653](https://arxiv.org/abs/2607.20653)，2026，Haocheng Yin\* / Shuohan Tao\* / Yongsheng Chen / Lu Gan · **佐治亚理工学院（Georgia Institute of Technology）**）把可变形操纵的世界模型写成 **可微 MPM 物理骨架 + 两个前馈网络**：MfM 从短窗视觉运动推断每粒子材料，RfD 在模拟器内部吸收接触 / 摩擦 / 本构简化带来的系统误差。

## 一句话定义

**保留可微 MPM 的物理结构，用「材料从运动」推断物体专属参数，再用有界残差修正解析动力学盖不住的 sim-to-real 偏差。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PhysCoRe | Physics-Corrected Residual World Model | 本文物理修正残差世界模型 |
| MPM | Material Point Method | 可变形连续介质粒子–网格离散 |
| MLS-MPM | Moving Least Squares MPM | 本文可微仿真骨干（+ APIC） |
| MfM | Material from Motion | 从观测运动推断每粒子材料与置信度 |
| RfD | Residual from Dynamics | 在 grid update↔G2P 间修正节点速度 |
| APIC | Affine Particle-in-Cell | 粒子–网格动量传递 |
| CD | Chamfer Distance | 几何预测误差主指标 |
| 3DGS | 3D Gaussian Splatting | 置信度可视化载体之一 |

## 为什么重要

- **物理混合族样本：** 在 [物理保真度输出轴](../overview/world-model-physics-fidelity-outputs.md) 上，相对纯像素 WM（[Masked Visual Actions](./paper-masked-visual-actions.md)）与触觉 WAM（[VT-WAM](./paper-vt-wam-visuotactile-contact-rich.md)），本工作显式保留 **本构 + 接触边界条件**。
- **替代逐物体优化：** 传统系统辨识慢、难泛化；MfM **单次前向** 且可在线适应未见物体。
- **置信度可行动：** 每粒子置信度与实际变形区域对齐，可作未来 **主动探索** 信号（KUKA 探测实验）。
- **分工清晰：** Table 4 显示 RfD 在弹塑性上额外 CD 收益更大——材料归 MfM，残差归 RfD。

## 核心信息

| 项 | 内容 |
|----|------|
| 机构 | 佐治亚理工学院（Georgia Tech） |
| 问题 | 可变形体在机器人 / 人手作用下的未来几何与运动预测 |
| 状态 | 粒子位置 / 速度 / 变形梯度 + 每粒子材料 \(\phi_p\) |
| 动作 | 相机帧级控制器速度，Catmull-Rom 上采样到 MPM 子步 |
| 骨干 | 可微 MLS-MPM；Fixed Corotated 或 von Mises Plasticity |
| 开源 | **未开源**（2026-07-27 核查） |

## 核心原理（方法）

### 问题设定

给定观测到的可变形物体与控制器点轨迹短窗，预测后续形变。材料 \(\phi_p=(\log E_p,\nu_p)\) 不可从单帧 RGB-D 直接读出，必须从运动中推断，并在未见构型上保持物理一致。

### PhysCoRe 双模块

| 模块 | 输入 → 输出 | 作用 |
|------|-------------|------|
| **MfM** | 短窗轨迹 Fourier 特征 → Graph U-Net | 每粒子 \(\phi_p\)、置信度 \(c_p\)、本构分支概率 \(\pi\) |
| **可微 MPM** | \(\phi\) + 边界速度（夹爪/指尖）+ 地面摩擦 | 保留 P2G / grid / G2P 物理循环 |
| **RfD** | 局部网格状态 + MfM 材料 + 接触几何 | 有界 \(\Delta\mathbf{v}_i\)（\(\tanh\) 限幅，零初始化） |

材料范围：\(\log E\in[5,11]\)，\(\nu\in[0.05,0.45]\)。MfM 在仿真增广数据上预训练；随后 **冻结 MfM**，在真机 episode 上端到端训 RfD（按 \(K\) 帧窗口刷新材料，每 \(H_r\) 子步施加残差）。

### 流程总览

```mermaid
flowchart LR
  OBS[短窗视觉运动] --> MfM
  MfM --> PHI["φ_p, c_p, π"]
  PHI --> MPM[可微 MLS-MPM]
  ACT[控制器 / 接触 BC] --> MPM
  MPM --> GRID[网格速度更新]
  GRID --> RfD
  RfD --> DV["有界 Δv"]
  DV --> G2P[G2P + 粒子更新]
  G2P --> PRED[未来形变 / 几何]
```

## 源码运行时序图

**不适用。** 截至 **2026-07-27** 开源核查：arXiv / HTML **无官方代码、项目页或权重**；GitHub 同名仓库与本文无关。可复述顺序训练（MfM 仿真预训练 → RfD 真机微调），但 **无可运行官方入口**。

## 实验要点（索引级）

| 轴 | 报告口径（以论文为准） |
|----|------------------------|
| 真机数据 | **12** episode：绳 / 毛巾 / 毛绒（弹性）+ Play-Doh（弹塑性）；抬、推、拉、挤 |
| 协议 | 前 **50%** 识别材料，后半段固定材料做未来预测 |
| vs PhysTwin | 弹性 CD **−43.7%**，弹塑性 **−30.5%**；tracking loss **−15.2% / −8.3%** |
| RfD 消融 | 相对 MfM-only：弹性 CD **−13.1%**，弹塑性 **−17.8%** |
| 置信度 | 随观测变形区域升高；KUKA 主动探测后局部置信度上升 |
| 指标 | CD、tracking loss、IoU、PSNR 等（见表） |

## 工程实践

| 项 | 实践要点 |
|----|----------|
| 何时选 | 需要 **材料可解释 + 物理结构** 的可变形预测 / 规划先验 |
| 训练顺序 | 先 MfM（仿真 GT 材料）再 RfD（真机，MfM 冻结） |
| 残差安全 | 输出层零初始化 + \(\delta_{\max}\) 限幅，避免破坏 rollout 稳定 |
| 主动探索 | 用低置信度区域引导下一次接触（论文展示信号，非完整主动学习系统） |
| 与像素 WM | 不替代视频 WM 的外观先验；可作接触丰富任务的 **动力学层** |
| 开源 | **未开源** — 选型时勿假设可 clone 复现 |

## 结论

**PhysCoRe 证明「可微物理 + 前馈材料推断 + 有界残差」能在真机可变形序列上同时提升几何精度并保留物理结构；材料置信度是比单一标量 loss 更可行动的诊断。**

1. **MfM 替代逐物体标定** — 短窗运动 → 每粒子弹性模量 / 泊松比，可迁移未见物体。
2. **RfD 专吃残差** — 弹塑性收益更大，说明解析本构缺口集中在接触与塑性。
3. **相对 PhysTwin 大幅降 CD** — 弹性 **−43.7%**、弹塑性 **−30.5%**（论文 Table 2）。
4. **置信度对齐真实变形** — 可作未来 confidence-guided exploration 的天然信号。
5. **边界清晰** — 当前主要覆盖弹性 / 弹塑性；撕裂、切割、流体、强粘附未声明支持。
6. **复现门槛** — 截至 2026-07-27 **无可运行官方代码**；工程落地需自研或等待发布。

## 局限与风险

- **材料行为集合封闭：** 预定义弹性 / 弹塑性分支，复杂接触与拓扑变化外推风险高。
- **数据规模小：** 12 段人手 episode，统计方差与跨操作者泛化需谨慎解读。
- **未开源：** 无法第三方审计实现细节与超参；与像素 WM 的联合闭环未展示。
- **勿与「画面物理」混淆：** 本页指标是 **3D 几何 / 跟踪**，不是 FVD；对照诊断见 [Imagined Rollouts…](./paper-imagined-rollouts-kinematic-not-dynamic.md)。

## 与相邻工作的对比（分界）

| 对比轴 | PhysCoRe | 纯学习粒子/高斯 WM | [VT-WAM](./paper-vt-wam-visuotactile-contact-rich.md) |
|--------|----------|-------------------|-----------------------------------------------------|
| **物理骨架** | **可微 MPM** | 弱 / 无 | 无显式 MPM |
| **材料** | **MfM 显式 \(\phi_p\)** | 隐式 | 触觉形变通道 |
| **残差** | **RfD 网格速度** | 端到端 | 流匹配联合生成 |
| **开源** | **未开源** | 视具体工作 | 有项目页 |

## 关联页面

- [世界模型物理保真度输出轴](../overview/world-model-physics-fidelity-outputs.md) — 「物理混合」族入口
- [Generative World Models](../methods/generative-world-models.md) — 像素域对照
- [运动学可行与动力学可行](../concepts/kinematic-vs-dynamic-feasibility.md) — 动力学约束语义
- [物理保真度与 Sim2Real 差距](../concepts/physics-fidelity-sim2real-gap.md) — sim-to-real 残差动机
- [VT-WAM](./paper-vt-wam-visuotactile-contact-rich.md) — 接触丰富另一信号通路
- [KineBench](./paper-kinebench.md) — 可执行性评测（刚体操纵侧）
- [CoRe（人形重定向，同名消歧）](./paper-core.md) / [CoRe 软件](./core-retarget.md)

## 参考来源

- [PhysCoRe 论文归档（arXiv:2607.20653）](../../sources/papers/physcore_arxiv_2607_20653.md)
- [具身智能研究室 · 世界模型物理保真度导读](../../sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)

## 推荐继续阅读

- [arXiv:2607.20653](https://arxiv.org/abs/2607.20653) — 论文全文
- [物理保真度输出轴](../overview/world-model-physics-fidelity-outputs.md) — 四类测试优先序
- [Imagined Rollouts are Kinematic, Not Dynamic](./paper-imagined-rollouts-kinematic-not-dynamic.md) — 动力学敏感性诊断对照
