---
type: entity
tags: [repo, unitree, unitreerobotics, imitation-learning, lerobot, teleoperation, humanoid]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./lerobot.md
  - ./xr-teleoperate.md
  - ./unitree-sim-isaaclab.md
  - ./unitree-dexterous-hand-services.md
  - ./unifolm-vla.md
  - ../methods/imitation-learning.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/repos/unitree_lerobot.md
  - ../../sources/repos/unitree.md
summary: "unitree_lerobot 是宇树基于 Hugging Face LeRobot 的官方改版，支持 G1 双臂灵巧手数据转换、训练验证与真机推理；对接 xr_teleoperate 采数与 unitree_sim_isaaclab 仿真回放。"
---

# unitree_lerobot

**unitree_lerobot** 把 [LeRobot](./lerobot.md) 训练栈接到 Unitree **G1 + 灵巧手** 数据流：数据转换、策略训练验证、仿真回放与真机推理。

## 一句话定义

官方模仿学习「训练与部署」胶水仓——输入遥操作/仿真采集的数据，输出可在 G1 上验证的 LeRobot 策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 模仿学习 |
| LeRobot | Hugging Face LeRobot | 开源机器人学习框架与数据集格式 |
| VLA | Vision-Language-Action | 视觉–语言–动作；可与 UnifoLM 衔接 |
| G1 | Unitree G1 Humanoid | 目标人形平台 |
| DDS | Data Distribution Service | 真机/仿真通信 |
| XR | Extended Reality | 采数常用前端 |

## 为什么重要

- 打通 **采数 → 数据集格式 → 训练 → 真机** 的官方叙述，避免每人自写一份 LeRobot 适配。
- v0.3 起对齐 **LeRobot dataset v3.0**，并扩展 `pi05` / `groot` 等策略支持（以上游 Release Note 为准）。
- 与 [`xr_teleoperate`](./xr-teleoperate.md)、[`unitree_sim_isaaclab`](./unitree-sim-isaaclab.md) 组成人形 IL 闭环。

## 核心原理

| 目录 | 职责 |
|------|------|
| `lerobot/` | 内嵌/固定 commit 的训练代码（README 标注对应上游 commit） |
| `utils/` | Unitree 数据转换与处理 |
| `eval_robot/` | 真机推理验证 |

**版本能力摘要**：

- v0.1：G1 + Dex1 + Dex3 的转换 / 部署 / 真机测试
- v0.2：Brainco / Inspire 手、数据集回放、仿真验证
- v0.3：dataset v3.0、更多 policy

## 工程实践

```bash
git clone --recurse-submodules https://github.com/unitreerobotics/unitree_lerobot.git
# 先按 LeRobot 官方指南装依赖，再跟本仓文档做 Unitree 数据转换与训练
```

关联资源：Hugging Face [`unitreerobotics`](https://huggingface.co/unitreerobotics) 数据集；仿真侧用 `unitree_sim_isaaclab`；灵巧手桥接见 [灵巧手 Serial↔DDS 服务](./unitree-dexterous-hand-services.md)。

## 局限与风险

- **子模块与 commit 钉扎**：勿随意升级内嵌 lerobot 而不跑转换回归。
- 真机推理依赖正确的手部服务与 DDS 配置；手型（Dex1/Dex3/Inspire/Brainco）与数据转换脚本必须一致。
- 本仓是训练/验证改版，不等价于 UnifoLM 基础模型权重仓。

## 关联页面

- [LeRobot](./lerobot.md)
- [xr_teleoperate](./xr-teleoperate.md)
- [unitree_sim_isaaclab](./unitree-sim-isaaclab.md)
- [UnifoLM-VLA](./unifolm-vla.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_lerobot.md](../../sources/repos/unitree_lerobot.md)
- 上游：<https://github.com/unitreerobotics/unitree_lerobot>

## 推荐继续阅读

- 仓内 `docs/README_zh.md`

