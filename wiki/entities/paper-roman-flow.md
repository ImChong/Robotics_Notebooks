---
type: entity
tags: [paper, offline-rl, normalizing-flow, manipulation, diffusion-policy]
status: complete
updated: 2026-08-22
arxiv: "2608.20208"
code: https://github.com/konnyaku28/RoMAN-Flow
related:
  - ../comparisons/online-vs-offline-rl.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../entities/libero-benchmark.md
  - ../overview/video-contact-control-10-papers-technology-map.md
sources:
  - ../../sources/papers/roman_flow_arxiv_2608_20208.md
  - ../../sources/repos/roman-flow.md
  - ../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md
summary: "RoMAN-Flow（arXiv:2608.20208）：AR-NF 离线 RL + sampling-free advantage-weighted likelihood + 一步蒸馏；LIBERO/RoboMimic 已开源可复现。"
---

# RoMAN-Flow

**RoMAN-Flow: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation**（[arXiv:2608.20208](https://arxiv.org/abs/2608.20208)，[代码](https://github.com/konnyaku28/RoMAN-Flow)）——（见论文作者列表）。

## 一句话定义

**可计算似然让离线 RL 有直接抓手——训练不采样、部署一步生成。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AR-NF | Autoregressive Normalizing Flow | 自回归归一化流策略 |
| IQL | Implicit Q-Learning | 离线 RL 后训练阶段 |
| IL | Imitation Learning | 行为克隆预训练阶段 |

## 为什么重要

- 纳入 [具身智能小站 2026-08-22 十篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md) 的「视频→接触→控制→VLA 持续学习」主线。
- 开源状态（入库日）：**已开源**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | （见论文作者列表） |
| **出处** | arXiv:2608.20208（2026-08） |
| **开源** | **已开源** |

### 流程总览

```mermaid
flowchart LR
  data[离线演示] --> il[IL 预训练 AR-NF]
  il --> iql[IQL advantage-weighted likelihood]
  iql --> distill[一步 BiFlow 蒸馏]
  distill --> deploy[低延迟部署]
```

## 结论

**离线操作要同时保住似然可处理性与部署延迟，AR-NF + 蒸馏是可行折中。**

- 优化阶段 sampling-free，避免从自回归策略采样
- 蒸馏后一步动作生成显著降推理延迟
- 官方仓含 LIBERO-10/Long 与 RoboMimic 全流程
- HF 权重与 manifest 外部分发

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Setup as setup_env.sh
    participant Buf as prepare_*_buffer.py
    participant Train as main_torch.py
    participant Eval as evaluate.py / eval_biflow_torch.py
    Dev->>Setup: 安装 LIBERO/RoboMimic 依赖
    Dev->>Buf: 构建 Zarr 训练/评测 buffer
    Dev->>Train: IL → IQL → One-Step 蒸馏
    Train-->>Dev: params_*.pt + flags.json
    Dev->>Eval: 加载 WEIGHTS_ROOT 发布 checkpoint
    Eval-->>Dev: LIBERO / RoboMimic CSV 成功率
```

## 与其他页面的关系

- [offline-rl](../comparisons/online-vs-offline-rl.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [libero-benchmark](../entities/libero-benchmark.md)
- [视频–接触–控制 10 篇技术地图](../overview/video-contact-control-10-papers-technology-map.md)

## 参考来源

- [roman_flow_arxiv_2608_20208](../../sources/papers/roman_flow_arxiv_2608_20208.md)
- [roman-flow](../../sources/repos/roman-flow.md)
- [wechat_embodied_station_video_contact_control_10_papers_2026-08-22](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)

## 推荐继续阅读

- [arXiv:2608.20208](https://arxiv.org/abs/2608.20208)
- [RoMAN-Flow 官方代码](https://github.com/konnyaku28/RoMAN-Flow)
