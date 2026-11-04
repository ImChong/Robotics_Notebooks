---
type: entity
project_id: quadruped-pympc
tags:
  - repo
  - iit
  - quadruped
  - mpc
  - acados
  - jax
  - mujoco
status: complete
updated: 2026-11-01
code: https://github.com/iit-DLSLab/Quadruped-PyMPC
related:
  - ../methods/model-predictive-control.md
  - ../methods/nonlinear-model-predictive-control.md
  - ../entities/acados.md
  - ../methods/mppi.md
  - ../tasks/locomotion.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/quadruped_pympc.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "Quadruped-PyMPC（IIT DLS Lab）是基于单刚体模型、用 Python 写的 MPC：梯度式（acados）与采样式（JAX）两种实现，已在真机测试并兼容 MuJoCo。"
---

# Quadruped-PyMPC

**Quadruped-PyMPC**（[GitHub](https://github.com/iit-DLSLab/Quadruped-PyMPC)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「优化控制（MPC / WBC）」组。

## 一句话定义

**Quadruped-PyMPC（IIT DLS Lab）是基于单刚体模型、用 Python 写的 MPC：梯度式（acados）与采样式（JAX）两种实现，已在真机测试并兼容 MuJoCo。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 模型预测控制 |
| SRBD | Single Rigid Body Dynamics | 单刚体动力学 |
| RTI | Real-Time Iteration | 实时迭代 |

## 为什么重要

- 综述把它放在「优化控制（MPC / WBC）」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：446（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [iit-DLSLab/Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC) |
| **最近提交** | 2026-10-15 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 优化控制（MPC / WBC） |

## 文章说法核对

- **文章：** 文章称它提供基于 Python 的 MPC 与 WBC 实现，用 acados 求解器，适合学习。
- **README / 核查：** README：梯度式 MPC 在 i7-13700H 上计算不到 5 ms，可选积分器、足端力平滑、落脚点优化、实时迭代等；采样式 MPC 在 NVIDIA 4050 mobile GPU 上 10000 并行 rollout 不到 2 ms。
- **差异与保留：** 文章说 WBC 实现；README 前段只见 MPC 两种实现，WBC 未在已读段落出现，未核实。

## 源码运行时序图

入口：`python3 simulation/simulation.py`（`__main__` 读取 `quadruped_pympc/config.py` 后调用 `run_simulation`；MuJoCo 环境来自外部依赖 `gym_quadruped`）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant SIM as simulation/simulation.py<br/>run_simulation
    participant ENV as gym_quadruped.QuadrupedEnv<br/>(MuJoCo 仿真)
    participant WR as quadruped_pympc_wrapper.py<br/>QuadrupedPyMPC_Wrapper
    participant WBI as interfaces/wb_interface.py<br/>WBInterface
    participant SRBD as interfaces/srbd_controller_interface.py<br/>SRBDControllerInterface
    participant CTL as controllers/ (梯度式 acados / 采样式 JAX)

    U->>SIM: python3 simulation/simulation.py
    SIM->>ENV: QuadrupedEnv(robot, scene, sim_dt, ...) 并 reset()
    SIM->>WR: QuadrupedPyMPC_Wrapper(初始足端位置, legs_order, ...)
    WR->>SRBD: 创建 SRBDControllerInterface（按 mpc_params type 选控制器）
    WR->>WBI: 创建 WBInterface（步态、足端参考、摆动、IK 等子模块）

    loop 每个仿真步（env.step_num）
        SIM->>ENV: 读取足端/基座/质心状态、雅可比、质量矩阵、接触
        SIM->>WR: compute_actions(com, base, feet, qpos, qvel, jacobians, contact, ...)
        WR->>WBI: update_state_and_reference(...)
        Note over WBI: 地形估计、速度调制、步态相位与接触序列、足端参考（可选视觉落脚点适应）
        WBI-->>WR: state_current, ref_state, contact_sequence, optimize_swing

        alt 到达 MPC 求解周期（step_num 按 mpc_frequency 取模）
            WR->>SRBD: compute_control(state, ref, contact_sequence, inertia, phase, step_freq, ...)
            SRBD->>CTL: 求解 SRBD 最优控制（梯度式 NMPC 或采样式 rollout）
            opt use_RTI 为真（梯度式）
                WR->>SRBD: compute_RTI() 预先线性化
            end
            opt optimize_step_freq 为真（非采样式）
                WR->>WR: 调用 srbd_batched_controller_interface.optimize_gait()
            end
            CTL-->>SRBD: GRFs, 落脚点, 关节参考
            SRBD-->>WR: nmpc_GRFs, nmpc_footholds, ...
        end

        WR->>WBI: compute_stance_and_swing_torque(dt, qpos, qvel, jacobians, ...)
        Note over WBI: 摆动段用 SwingTrajectoryController 做笛卡尔控制；关节参考由 InverseKinematicsNumeric 求解
        WBI-->>WR: tau, des_joints_pos, des_joints_vel
        WR-->>SIM: tau（各腿力矩）
        SIM->>SIM: 按 0.9 × actuator_ctrlrange 对 tau 限幅
        SIM->>ENV: env.step(action)
        SIM->>WR: get_obs() 记录控制器观测量
    end
```

图依据：iit-DLSLab/Quadruped-PyMPC@c46cb9a，实际读过 `README.md`、`README_install.md`、`simulation/simulation.py`、`quadruped_pympc/config.py`（前段）、`quadruped_pympc/quadruped_pympc_wrapper.py`、`quadruped_pympc/interfaces/srbd_controller_interface.py`、`quadruped_pympc/interfaces/wb_interface.py`；未运行代码。

## 关联页面

- [model-predictive-control](../methods/model-predictive-control.md)
- [nonlinear-model-predictive-control](../methods/nonlinear-model-predictive-control.md)
- [acados](../entities/acados.md)
- [mppi](../methods/mppi.md)
- [locomotion](../tasks/locomotion.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [quadruped_pympc.md](../../sources/repos/quadruped_pympc.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
- 源码核对（2026-11-02）：iit-DLSLab/Quadruped-PyMPC@c46cb9a，读过 README.md、README_install.md、simulation/simulation.py、quadruped_pympc/config.py（前段）、quadruped_pympc/quadruped_pympc_wrapper.py、quadruped_pympc/interfaces/srbd_controller_interface.py、quadruped_pympc/interfaces/wb_interface.py；时序图依据以上文件，未运行代码。
