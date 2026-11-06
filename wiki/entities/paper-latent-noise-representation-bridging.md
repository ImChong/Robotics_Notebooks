---
type: entity
project_id: latent-noise-representation-bridging
tags:
  - paper
  - manipulation
  - imitation-learning
status: complete
updated: 2026-11-06
arxiv: "2611.07275"
code: https://github.com/Maincode-Research/LARBRID
related:
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-06-20-papers-technology-map.md
sources:
  - ../../sources/papers/latent_noise_representation_bridging_arxiv_2611_07275.md
  - ../../sources/repos/latent_noise_representation_bridging.md
  - ../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md
summary: "Latent Noise Representation Bridging（arXiv:2611.07275）：同一动作可能由不同噪声生成，直接给噪声打分容易学偏。"
---

# Latent Noise Representation Bridging（arXiv:2611.07275）

**Latent Noise Representation Bridging**（*Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging*，[arXiv:2611.07275](https://arxiv.org/abs/2611.07275)，[代码](https://github.com/Maincode-Research/LARBRID)）来自 [具身智能小站 2026-11-06 盘点](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)，属「操作策略与数据」组。

## 一句话定义

**同一动作可能由不同噪声生成，直接给噪声打分容易学偏。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Latent Noise Representation Bridging | Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging | 本文方法 / 基准的简称 |

## 为什么重要

- 冻结扩散策略，从初始噪声调整动作。LARBRID 让噪声和动作共享价值表示，只调整初始噪声分布，在仿真运动与操作任务上检验适应效果。
- 开源结论：**已开源**（步骤 2.5，2026-11-06）。仓库可达（git ls-remote）。
- 同期同组工作见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.07275](https://arxiv.org/abs/2611.07275) |
| **开源** | **已开源** |
| **要点** | 同一动作可能由不同噪声生成，直接给噪声打分容易学偏。 |
| **文内指标** | 公众号转述：LARBRID 让噪声和动作共享价值表示，只调整初始噪声分布，在仿真运动与操作任务上检验适应效果。 |

## 源码运行时序图

以下为按仓库入口（`Maincode-Research/LARBRID`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Maincode-Research/LARBRID
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Latent Noise Representation Bridging 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：LARBRID 让噪声和动作共享价值表示，只调整初始噪声分布，在仿真运动与操作任务上检验适应效果。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**冻结扩散策略，从初始噪声调整动作：LARBRID 让噪声和动作共享价值表示，只调整初始噪声分布，在仿真运动与操作任务上检验适应效果。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：同一动作可能由不同噪声生成，直接给噪声打分容易学偏。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [imitation-learning](../methods/imitation-learning.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-06-20-papers-technology-map](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)

## 参考来源

- [latent_noise_representation_bridging_arxiv_2611_07275.md](../../sources/papers/latent_noise_representation_bridging_arxiv_2611_07275.md)
- [wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)
- [arXiv:2611.07275](https://arxiv.org/abs/2611.07275)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.07275)
- [https://github.com/Maincode-Research/LARBRID](https://github.com/Maincode-Research/LARBRID)
