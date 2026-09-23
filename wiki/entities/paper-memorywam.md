---
type: entity
tags:
  - paper
  - world-model
  - wam
  - memory
  - long-horizon
status: complete
updated: 2026-09-23
arxiv: "2606.20562"
code: https://github.com/yangsizhe/MemoryWAM
related:
  - ../concepts/world-action-models.md
  - ./paper-fast-wam.md
  - ./paper-tempowam.md
  - ./lingbot-vla.md
  - ../overview/embodied-frontier-algorithms-technology-map.md
sources:
  - ../../sources/papers/memorywam_arxiv_2606_20562.md
  - ../../sources/repos/memorywam.md
  - ../../sources/sites/memorywam.md
  - ../../sources/blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md
summary: "MemoryWAM（arXiv:2606.20562）：混合持久记忆 WAM：滑窗近期帧 + 任务起点 anchor + gist token 压缩长历史；推理复杂度 O(N)→O(N/d)，长时序家务优于 VLA/WAM 基线。"
---

# MemoryWAM（arXiv:2606.20562）

**MemoryWAM**（*MemoryWAM: Efficient World Action Modeling with Persistent Memory*，[arXiv:2606.20562](https://arxiv.org/abs/2606.20562)，[项目页](https://yangsizhe.github.io/MemoryWAM/)，[代码](https://github.com/yangsizhe/MemoryWAM)）来自 [机器人研发工程师 · 前沿算法盘点](../../sources/blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md)。

## 一句话定义

**混合持久记忆 WAM：滑窗近期帧 + 任务起点 anchor + gist token 压缩长历史；推理复杂度 O(N)→O(N/d)，长时序家务优于 VLA/WAM 基线。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MemoryWAM | Memory World Action Model | 本文带持久记忆的 WAM |
| WAM | World Action Model | 世界–动作联合模型 |
| KV | Key-Value Cache | 注意力键值缓存 |
| MoT | Mixture-of-Transformers | 多专家 Transformer |

## 为什么重要

- 全历史 KV 缓存贵；纯滑窗在非 Markov 家务任务上丢进度。
- 开源结论：**已开源**（步骤 2.5，2026-09-23）。
- 与 [具身前沿算法技术地图](../overview/embodied-frontier-algorithms-technology-map.md) 同路线条目可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2606.20562](https://arxiv.org/abs/2606.20562) |
| **开源** | **已开源** |
| **要点** | MoT 视频 DiT + 动作 DiT；gist token 蒸馏长程；推理跳过像素生成只更新 KV。 |
| **文内指标** | RMBench 等长时序记忆依赖任务（作者报告 83.0% 均值等）。 |


## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户/评测脚本
    participant R as MemoryWAM 仓库
    participant M as 模型权重
    participant E as 仿真/真机环境
    U->>R: clone + 依赖安装（见 README）
    U->>M: 下载 checkpoint（HF/Release）
    U->>R: train / eval 入口
    R->>E: rollout / 指标日志
    E-->>U: success / latency 等
```

图下说明：复现以 [`sources/repos/memorywam.md`](../../sources/repos/memorywam.md) 与官方 README 为准。


## 实验与评测

- RMBench 等长时序记忆依赖任务（作者报告 83.0% 均值等）。
- **读法：** 索引级摘要；逐项 baseline 以原文 PDF 为准。

## 与其他工作对比

| 维度 | MemoryWAM（本页） | [Fast-WAM](./paper-fast-wam.md) | 定长滑窗 WAM / VLA |
|------|--------------------|----------------------------------|---------------------|
| 动的旋钮 | **长历史怎么留**（滑窗 + 任务起点 anchor + gist token） | **推理期要不要想象未来** | 不动，靠窗口长度硬扛 |
| 推理复杂度 | $O(N)\to O(N/d)$（gist 压缩） | 去掉未来去噪，报告 190 ms | 随窗口线性增长 |
| 主攻任务 | **长时序、记忆依赖** 的家务任务 | 低延迟闭环 | 短时序任务 |
| 典型失效 | gist 压缩丢掉了后期才用得上的细节 | 训练期若也省掉视频则结论不成立 | 超出窗口即失忆 |

- **两者不是竞争关系：** MemoryWAM 解决「记得住」，Fast-WAM 解决「跑得快」，同属 [World-Action Models](../concepts/world-action-models.md) 谱系的不同旋钮；把它们的成功率放在一张表里比大小，比的是任务集不是方法。
- **anchor 的作用常被忽略：** 任务起点帧被单独保留，是因为长时序家务里 **初始场景状态** 决定了成功判据（东西原来在哪），滑窗一旦滚过去就再也找不回来。
- **数值口径：** RMBench 等长时序任务上 **83.0% 均值（作者报告）**，逐项 baseline **以 [原文 PDF](https://arxiv.org/abs/2606.20562) 为准**；本页为索引级摘要。

## 结论

**MemoryWAM 代表 WAM 记忆结构工程化；与 TempoWAM 执行层、Fast-WAM 延迟优化可组合阅读。**

1. 开源边界：**已开源** — 以项目页/仓库实际链接为准（入库日 2026-09-23）。
2. 核心机制：MoT 视频 DiT + 动作 DiT；gist token 蒸馏长程；推理跳过像素生成只更新 KV。…
3. 部署前核对硬件栈与评测协议，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [paper-fast-wam](./paper-fast-wam.md)
- [paper-tempowam](./paper-tempowam.md)
- [lingbot-vla](./lingbot-vla.md)

## 参考来源

- [memorywam_arxiv_2606_20562.md](../../sources/papers/memorywam_arxiv_2606_20562.md)
- [wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md](../../sources/blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md)
- [arXiv:2606.20562](https://arxiv.org/abs/2606.20562)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2606.20562)
- [项目页](https://yangsizhe.github.io/MemoryWAM/)
- [代码](https://github.com/yangsizhe/MemoryWAM)

