---
type: entity
tags: [repo, humanoid, loco-manipulation, sim2sim, mujoco, unitree-g1, omnicontact, noitom]
status: complete
updated: 2026-07-01
related:
  - ./paper-omnicontact-humanoid-loco-manipulation.md
  - ./unitree-g1.md
  - ../concepts/sim2real.md
  - ./humanoid-gym.md
  - ./visualmimic.md
sources:
  - ../../sources/repos/omnicontact-sim2sim.md
summary: "OmniContact_sim2sim：OmniContact 官方 MuJoCo 部署栈，CF-Gen 参考生成 + CF-Track ONNX 策略，支持单 skill、skill chaining、NPZ 轨迹跟踪与 Xbox FSM 热切换。"
---

# OmniContact sim2sim

**OmniContact_sim2sim**（[GitHub](https://github.com/Ingrid789/OmniContact_sim2sim)）是 [OmniContact](./paper-omnicontact-humanoid-loco-manipulation.md)（arXiv:2606.26201）的 **MuJoCo sim2sim** 官方实现：**CFgen** 生成任务空间 contact-flow 参考，**CFtrack** ONNX 策略执行跟踪。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CF | Contact Flow | 稀疏体目标 + 四端接触信号的 skill 接口 |
| FSM | Finite State Machine | Passive→DefaultPose→LocoMode→OmniContact 状态链 |
| NPZ | NumPy Archive | G1 重定向人–物轨迹文件格式 |
| ONNX | Open Neural Network Exchange | 导出的 CF-Track 策略推理格式 |

## 核心能力

| 路径 | 脚本 | 用途 |
|------|------|------|
| 脚本化 | `run_skill_omnicontact.py` | `--reference-source CFgen` 或 `NPZmotion` |
| 交互式 | `deploy_omnicontact.py` | Xbox 手柄热切换，镜像 sim2real FSM |

支持单 skill（`carrybox`、`pushbox`、`kickball`…）、链式 preset（`carry-push`、`carryheart`…）及 `data/` 下 NPZ 全轨迹回放。

## 关联页面

- [OmniContact（论文）](./paper-omnicontact-humanoid-loco-manipulation.md)
- [Unitree G1](./unitree-g1.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [sources/repos/omnicontact-sim2sim.md](../../sources/repos/omnicontact-sim2sim.md)
- <https://github.com/Ingrid789/OmniContact_sim2sim>

## 推荐继续阅读

- [OmniContact 项目页 MuJoCo WASM demo](https://omnicontact.github.io/)
