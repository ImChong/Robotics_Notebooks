---
type: overview
tags: [wechat-curator, microduck, pollen-robotics, sim2real, ppo, onnx, rust-runtime, tutorial]
status: complete
updated: 2026-10-02
related:
  - ./zhixing-microduck-primer-part1-zero-hardware-start.md
  - ./zhixing-microduck-primer-part2-play-without-code.md
  - ./zhixing-microduck-primer-part3a-cloud-gpu-ppo-training.md
  - ./zhixing-microduck-primer-part3b-ppo-logs-onnx-export.md
  - ./zhixing-microduck-primer-part4-rust-runtime-onnx.md
  - ../entities/pollen-microduck.md
  - ../entities/pollen-microduck-rl.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md
  - ../../sources/blogs/wechat_zhixing_microduck_primer_part1_zero_hardware_2026-10-02.md
  - ../../sources/blogs/wechat_zhixing_microduck_primer_part2_play_no_code_2026-10-02.md
  - ../../sources/blogs/wechat_zhixing_microduck_primer_part3a_cloud_gpu_ppo_2026-10-02.md
  - ../../sources/blogs/wechat_zhixing_microduck_primer_part3b_ppo_logs_onnx_2026-10-02.md
  - ../../sources/blogs/wechat_zhixing_microduck_primer_part4_rust_runtime_onnx_2026-10-02.md
summary: "智践行微信专辑 5 篇 Microduck 入门链：零硬件认知 → 仿真玩策略 → 云 GPU PPO → 日志与 ONNX → Rust mock 部署对齐；每篇独立 wiki 详情节点 + 官方两仓实体。"
---

# 智践行 · Microduck 具身智能入门专辑 — 技术地图

## 一句话定义

本页索引 **智践行** 公众号专辑（5 篇）的 **独立详情节点**：把 Pollen [Microduck RL](../entities/pollen-microduck-rl.md) 与 [Runtime](../entities/pollen-microduck.md) 的 sim2real 主线拆成可跟做的阅读顺序，算法与物理细节仍以官方仓库与实体页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | 训练仓默认 on-policy 算法 |
| ONNX | Open Neural Network Exchange | 跨端策略格式；61→14 维合同 |
| RL | Reinforcement Learning | 仿真交互学习策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| BAM | Better Actuator Models | XL330 电压执行器物理模型 |

## 专辑入口

| # | 标题 | 微信 | blog | wiki 详情 |
|---|------|------|------|-----------|
| 1 | 0 硬件起步 | [链接](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492291&idx=1&sn=853662478d99f4390605c3ec55078d59) | [part1](../../sources/blogs/wechat_zhixing_microduck_primer_part1_zero_hardware_2026-10-02.md) | [part1](./zhixing-microduck-primer-part1-zero-hardware-start.md) |
| 2 | 不写代码先玩 | [链接](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492360&idx=1&sn=dae6b7ed7716bfb6fdb2c78279cca5f6) | [part2](../../sources/blogs/wechat_zhixing_microduck_primer_part2_play_no_code_2026-10-02.md) | [part2](./zhixing-microduck-primer-part2-play-without-code.md) |
| 3上 | 云 GPU PPO | [链接](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492482&idx=1&sn=f2f318a943b8f8ae45a48a6907cabda6) | [part3a](../../sources/blogs/wechat_zhixing_microduck_primer_part3a_cloud_gpu_ppo_2026-10-02.md) | [part3a](./zhixing-microduck-primer-part3a-cloud-gpu-ppo-training.md) |
| 3下 | 日志与 ONNX | [链接](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492496&idx=1&sn=767310ed64bd0345a7dc42381f9fe597) | [part3b](../../sources/blogs/wechat_zhixing_microduck_primer_part3b_ppo_logs_onnx_2026-10-02.md) | [part3b](./zhixing-microduck-primer-part3b-ppo-logs-onnx-export.md) |
| 4 | Rust 运行时 | [链接](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492524&idx=1&sn=03dfcab89f09e598d946cb073c69ae5e) | [part4](../../sources/blogs/wechat_zhixing_microduck_primer_part4_rust_runtime_onnx_2026-10-02.md) | [part4](./zhixing-microduck-primer-part4-rust-runtime-onnx.md) |

## 流程总览

```mermaid
flowchart LR
  A[① 两仓认知] --> B[② infer/play 玩]
  B --> C[③上 云 GPU train]
  C --> D[③下 export + 日志]
  D --> E[④ Rust policy-rehearsal]
  E --> F[真机 / duck-sim 进阶]
```

## 为什么重要

- **5/5 独立节点**：每篇公众号对应唯一 wiki 页，避免与 [pollen-microduck](../entities/pollen-microduck.md) 实体重复叙事。
- **零硬件可走到 ④**：系列刻意在 Rust mock 闭合推理链，再谈奖励调参与真机。

## 关联页面

- [Pollen Microduck](../entities/pollen-microduck.md) — Runtime 与 daemon 边界
- [Microduck RL](../entities/pollen-microduck-rl.md) — 训练、BAM、61D 合同
- [Sim2Real](../concepts/sim2real.md) — 域随机化与执行器保真语境

## 参考来源

- [专辑 raw 归档](../../sources/raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md)
- [智践行 5 篇 blog 归档](../../sources/blogs/wechat_zhixing_microduck_primer_part1_zero_hardware_2026-10-02.md)

## 推荐继续阅读

- [microduck policy manifest（schema 2）](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md)
- [microduck_rl AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)
