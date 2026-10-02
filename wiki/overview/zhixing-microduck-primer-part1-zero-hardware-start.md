---
type: overview
tags: [microduck, pollen-robotics, tutorial, wechat-curator, sim2real, open-source]
status: complete
updated: 2026-10-02
related:
  - ./zhixing-microduck-primer-album-technology-map.md
  - ./zhixing-microduck-primer-part2-play-without-code.md
  - ../entities/pollen-microduck.md
  - ../entities/pollen-microduck-rl.md
  - ../entities/open-duck-mini.md
sources:
  - ../../sources/blogs/wechat_zhixing_microduck_primer_part1_zero_hardware_2026-10-02.md
  - ../../sources/raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md
  - ../../sources/sites/pollen-robotics-microduck.md
summary: "智践行专辑第①篇：零硬件建立 Microduck 两仓分工（microduck_rl 训 / microduck 部署）与 61 维策略合同，为后续仿真玩与云训练铺路。"
---

# 具身智能入门① · 零硬件 Microduck 认知起点

## 一句话定义

**不买整机** 也能从 Pollen 开源栈入门：先分清 **Python 训练仓** 与 **Rust 机载 Runtime**，理解 ONNX 是两者之间的唯一「桥」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 在 mjlab 仿真里学行走/技能 |
| ONNX | Open Neural Network Exchange | 导出给 Runtime 的策略格式 |
| MJCF | MuJoCo XML Format | 机器人仿真模型描述 |
| PPO | Proximal Policy Optimization | 默认训练算法 |
| BLE | Bluetooth Low Energy | 真机无网时 `duckctl` 通道 |

## 为什么重要

- 把 **商品机** 与 **开源软件** 拆开：学习 sim2real 不依赖预售到货。
- 与 [Open Duck Mini](../entities/open-duck-mini.md) 的 DIY 路线对照：Microduck 强调 **fork 官方训练+Runtime 合同**。

## 核心原理

### 两仓分工

| 组件 | 仓库 | 输出/输入 |
|------|------|-----------|
| 训练 | `pollen-robotics/microduck_rl` | checkpoint → **export.py** → ONNX |
| 部署 | `pollen-robotics/microduck` | `Policy::load(ONNX)` → 50 Hz `robotd` |

### 硬合同（先记住数字）

- **61 维观测** = 48 本体感觉 + 13 维命令块（twist / 头 / 身体姿态）；未用槽位 **零填充**。
- **14 维动作** = 除鸭嘴外舵机目标；与 [Microduck RL](../entities/pollen-microduck-rl.md) 关节表一致。

## 工程实践

1. 读 [产品页](https://pollen-robotics.com/microduck) 与 [Runtime README](https://github.com/pollen-robotics/microduck) 建立能力清单（走、滚、起身…）。
2. 浏览 [microduck_rl 任务表](https://github.com/pollen-robotics/microduck_rl#tasks) 看主任务 `Mjlab-Velocity-Flat-MicroDuck`。
3. 下一篇直接在仿真 **玩** 官方策略，无需写代码。

## 局限与风险

- 15 路硬件舵机 vs 14 路 RL 执行器：口径差见 [pollen-microduck](../entities/pollen-microduck.md) 实体页，勿混读为 bug。
- ① 仅为认知；**不能替代** smoke test 与 export 规范。

## 关联页面

- [专辑技术地图](./zhixing-microduck-primer-album-technology-map.md)
- [② 不写代码先玩](./zhixing-microduck-primer-part2-play-without-code.md)

## 参考来源

- [wechat_zhixing_microduck_primer_part1_zero_hardware_2026-10-02.md](../../sources/blogs/wechat_zhixing_microduck_primer_part1_zero_hardware_2026-10-02.md)

## 推荐继续阅读

- [Pollen Microduck 实体页](../entities/pollen-microduck.md)
