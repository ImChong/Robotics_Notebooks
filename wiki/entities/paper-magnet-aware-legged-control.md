---
type: entity
tags: [paper, quadruped, mpc, whole-body-control, disturbance-estimation, magnetic-field, mujoco, unitree-go2, tum]
status: complete
updated: 2026-10-12
arxiv: "2610.08653"
project_id: magnet-aware-legged-control
related:
  - ../overview/frontier-humanoid-quadruped-2026-10-05-10-09.md
  - ../tasks/locomotion.md
  - ./quadruped-robot.md
  - ../concepts/whole-body-control.md
  - ../concepts/mpc-wbc-integration.md
  - ../concepts/centroidal-dynamics.md
  - ./mujoco.md
sources:
  - ../../sources/papers/magnet_aware_control_arxiv_2610_08653.md
  - ../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md
summary: "arXiv:2610.08653：TUM 与 CERN 为 Unitree Go2 建立 MuJoCo 感应磁力插件、逆向磁场估计器及显式含磁旋量的质心 MPC 与 WBC；相对不补偿控制，可抵抗扰动力约 2.5 倍、力矩 1.6–2 倍，主要为仿真加硬件在环验证，代码未发布。"
---

# 腿式机器人的磁场感知控制

论文 *Magnet-Aware Control of Legged Robots*（[arXiv:2610.08653](https://arxiv.org/abs/2610.08653)）由 **慕尼黑工业大学（TUM）** 与 **欧洲核子研究中心（CERN）** 的作者完成：面向加速器、聚变装置等强磁场环境，让四足机器人把磁力当作可预测的位姿相关扰动，而不是盲目抑制的外力。

## 一句话定义

**给铁磁刚体建感应磁体模型，用逆向估计器从机器人自己的动力学响应里反推磁场，再把磁力与磁力矩直接加进 MPC 与全身控制的动力学方程。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 质心 MPC，磁力进入质心动力学 |
| WBC | Whole-Body Control | QP 形式，动力学含 Jmᵀ W_m 项 |
| HIL | Hardware-in-the-Loop | 磁旋量数值注入控制器，真机行走 |
| IMU | Inertial Measurement Unit | 提供基座加速度，用于场估计 |
| T | Tesla | 磁感应强度单位，论文中扰动场范围 0.02–1.0 T |

## 为什么重要

- 强磁场设施里人不便进入，却恰好需要机器人巡检；磁场既损坏传感器，又对电机等铁磁件施加力。
- 磁扰动随位姿和空间梯度变化，普通扰动观测器把它当任意外力，忽略了可预测结构。
- 该文把磁场作为场景先验与可估计量，给出“控制器可承受的安全区域”这类可量化指标。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 慕尼黑工业大学（TUM）；欧洲核子研究中心（CERN，BE/CEM/MRO） |
| **平台** | Unitree Go2（12 个 GO-M8018-6 电机，μr = 1000 最坏假设）；MuJoCo + 实验室 HIL |
| **传感** | 四个虚拟磁传感器（四面体布置，模拟 THM1176 霍尔传感器的噪声和偏置） |
| **开源** | 页面与正文无代码链接；截至 2026-10-12 未发布；未在 CERN 现场测试 |

## 方法栈 / 流程总览

```mermaid
flowchart LR
  plugin["MuJoCo 磁力插件\n感应磁矩 m = αB"]
  est["逆向磁场估计器\n优化 B0 与梯度 G"]
  mpc["磁感知质心 MPC\nAcados"]
  wbc["磁感知 WBC\nQP, W_m 线性化"]
  robot["Go2 行走"]
  plugin -->|"仿真与标定"| est
  robot -->|"关节力矩 + IMU + 磁传感"| est
  est -->|"B0, G"| mpc --> wbc --> robot
```

## 方法

- **磁模型：** 铁磁刚体按各向异性感应磁体处理：m = αB，α = (V/μ₀)χ_app，χ_app = χ(I + χN)⁻¹；磁旋量 W_m = [G(αB); (αB)×B]，其中 G 为场梯度。
- **MuJoCo 插件：** 任意刚体可标记为磁性体，设置相对磁导率、退磁因子与体积；在体质心处评估解析外场；提供线性场和球形场两种测试场。
- **场估计：** 扩展 Kolev & Todorov 的逆向估计，输入关节力矩、基座轨迹和传感器读数，前向仿真并优化 B₀ 与 G，最小化基座加速度误差与传感器读数误差；角加速度由陀螺仪低通有限差分得到，避开噪声大的关节加速度。
- **MPC：** 状态为归一化质心动量与构型，输入为地面反力与关节速度；磁力 F_m 与磁力矩 τ_m（含 r×F）直接进入质心动力学；Acados + CasADi + Pinocchio。
- **WBC：** qpOASES 的 QP，动力学为 M q̈ + h = Sᵀτ + Jcᵀλ + Jmᵀ W_m，W_m 局部一阶泰勒线性化，带力矩限幅。

## 源码运行时序图

**不适用** — 论文与 arXiv 页面没有代码仓库；仅有 Acados、Pinocchio 等依赖的公开文档，无法对齐本工作的入口脚本。

## 实验与评测

- **总体对比：** 相对不补偿系统，最大可抵抗扰动力提高 2.5 倍，扰动力矩提高 1.6–2 倍；基于仿真场图，安全工作区域增加 29.34%，其中 10.84% 此前会导致崩溃。
- **场估计（仿真）：** 机器人行走时估计收敛到范数 0.07 T 的静场；噪声或数据不足会影响收敛。
- **均匀偏航力矩：** 不补偿最多承受 6 N·m，偏航偏移约从 0.02 T 开始，0.1 T 附近稳态误差约 10°，更高场崩溃；补偿后可承受 10 N·m（约 0.18 T），饱和约 12 N·m。
- **球形梯度场（半径 2 m，中心场 [0.3,0.3,0] T）：** 不补偿在 0.02–0.03 T（约 15 N）失去平衡，0.05 T（约 20 N）必然失败；补偿后可承受峰值 50 N，通过 0.06 T 和 0.08 T 场，瞬态接近 1.0 T 时在执行器饱和前仍稳定。
- **航向误差：** 不补偿最大 5°，补偿后小于 1°。
- **HIL：** 磁旋量数值注入 WBC，真机行走；无量化数字，仅 Fig. 7 的定性改善。

## 结论

**把磁力当作可估计、可预测的位姿相关载荷比当作盲扰动更有效，但目前证据主要来自仿真，真实磁加载尚未验证。**

1. 真正的指标是可抵抗的扰动上限（力 2.5 倍、力矩 1.6–2 倍）和安全区域面积（+29.34%），不是跟踪误差。
2. 次要收益是航向误差由最大 5° 降到 1° 以内。
3. 补偿受限于硬件：力矩和摩擦极限一到，腿力超出摩擦时仍会倒。
4. 电机磁导率取纯铁最坏假设，等效磁矩不准会影响估计与补偿，部署前要标定。
5. HIL 是数值注入，不含结构载荷和关节卡滞，现场使用前须在受控磁体上做物理验证。
6. 29.34% 来自仿真场图，不能直接外推到 CERN 等真实设施。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [SRBD 凸 MPC + WBC](../concepts/srbd-convex-mpc-wbc.md) | 同类 MPC+WBC 分层结构，本文新增显式磁旋量项 |
| [MPC-WBC 集成](../concepts/mpc-wbc-integration.md) | 说明分层的接口；本文通过动力学约束注入扰动 |
| [四足本体力估计](./paper-proprioceptive-force-estimation-for-quadruped.md) | 通用外力估计；本文利用磁场结构估计少量参数（B₀、G） |
| [Feeling Through the Load](./paper-feeling-through-the-load-compliant-quadruped.md) | 学习式力估计；本文是模型化估计 |

## 局限与风险

- 仅线性与球形场；复杂场需更好的模型。
- 估计在噪声和有限传感下退化，仅仿真验证。
- 霍尔编码器换成电容编码器是假设，未测试。
- 开源状态：截至 2026-10-12 无仓库、无项目页；MuJoCo 插件未公开。

## 关联页面

- [人形与四足周报 2026-10-05 至 10-09](../overview/frontier-humanoid-quadruped-2026-10-05-10-09.md)
- [Locomotion 任务页](../tasks/locomotion.md)
- [全身控制](../concepts/whole-body-control.md)
- [MPC-WBC 集成](../concepts/mpc-wbc-integration.md)
- [质心动力学](../concepts/centroidal-dynamics.md)
- [MuJoCo](./mujoco.md)

## 参考来源

- [论文归档](../../sources/papers/magnet_aware_control_arxiv_2610_08653.md)
- [人形/四足周报来源](../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md)
- [arXiv:2610.08653](https://arxiv.org/abs/2610.08653)

## 推荐继续阅读

- [arXiv HTML 全文](https://arxiv.org/html/2610.08653v1)
- [Acados](https://docs.acados.org/)
- [MuJoCo 文档](https://mujoco.readthedocs.io/)
