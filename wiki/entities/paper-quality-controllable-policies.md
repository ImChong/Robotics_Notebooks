---
type: entity
project_id: quality-controllable-policies
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-10-28
arxiv: "2610.37393"
project: https://sinnnj.github.io/qcp
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/quality_controllable_policies_arxiv_2610_37393.md
  - ../../sources/sites/quality-controllable-policies.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "Quality-Controllable Policies（arXiv:2610.37393）：把速度、平滑性等质量指标作为策略条件，通过在线采样、成功轨迹回收与事后重标注迭代扩展可控范围。"
---

# Quality-Controllable Policies（arXiv:2610.37393）

**Quality-Controllable Policies**（*Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling*，[arXiv:2610.37393](https://arxiv.org/abs/2610.37393)，[项目页](https://sinnnj.github.io/qcp)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「VLA 与策略后训练」组。

## 一句话定义

**把速度、平滑性等质量指标作为策略条件，通过在线采样、成功轨迹回收与事后重标注迭代扩展可控范围。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Quality-Controllable Policies | Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling | 本文方法 / 基准的简称 |

## 为什么重要

- 成功之外，还要控制动作质量。在仿真及传送带、酒杯搬运真机任务中验证。
- 开源结论：**待发布**（步骤 2.5，2026-10-28）。项目页 HTTP 200，仅见 mujoco_menagerie 引用，页面含 soon 字样（脚本扫描）。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    D["成功演示数据集 D"] --> S["规则化质量评分 m(τ)<br/>如 1/T 速度、动作差分平滑度、关节路径长度"]
    S --> Q["按经验 CDF 等频量化为 B 个 bin，得质量标签 q(τ)（式 1）"]
    Q --> TR["训练质量条件策略 π_θ(A_t | h_t, q)<br/>flow matching，式 2 / 3"]
    TR --> INF["部署：用户指定目标 bin q⋆，积分速度场生成动作块"]
    subgraph IT["迭代 Hindsight Relabeling（§3.2，Algorithm 1）"]
        R1["用 π_θk 在最高 bin q⋆=B-1 下 rollout"] --> R2["只保留成功 rollout，计算其实际 m(τ)"]
        R2 --> R3["按实际得分事后重标，并入数据集 D_k+1"]
        R3 --> R4["用 D_k+1 重算经验 CDF 与全部轨迹标签"]
        R4 --> R5["重新初始化并从头训练 π_θk+1"]
        R5 --> R1
    end
    TR --> R1
    subgraph CD["条件注入方式"]
        C1["从头训练的小策略：标签嵌入向量与观测编码拼接"]
        C2["VLA（π0.5）：标签写入语言提示如 quality level: 3"]
    end
    C1 -.-> TR
    C2 -.-> TR
```

图依据：论文 §3.1 / §3.2 / Algorithm 1 / Fig.1

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.37393](https://arxiv.org/abs/2610.37393) |
| **开源** | **待发布** |
| **要点** | 把速度、平滑性等质量指标作为策略条件，通过在线采样、成功轨迹回收与事后重标注迭代扩展可控范围。 |
| **文内指标** | 公众号转述：在仿真及传送带、酒杯搬运真机任务中验证。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，仅见 mujoco_menagerie 引用，页面含 soon 字样（脚本扫描）。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：在仿真及传送带、酒杯搬运真机任务中验证。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**成功之外，还要控制动作质量：在仿真及传送带、酒杯搬运真机任务中验证。**

1. 开源边界：**待发布** — 项目页 HTTP 200，仅见 mujoco_menagerie 引用，页面含 soon 字样（脚本扫描）。
2. 核心机制：把速度、平滑性等质量指标作为策略条件，通过在线采样、成功轨迹回收与事后重标注迭代扩展可控范围。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [quality_controllable_policies_arxiv_2610_37393.md](../../sources/papers/quality_controllable_policies_arxiv_2610_37393.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.37393](https://arxiv.org/abs/2610.37393)
- [arXiv PDF](https://arxiv.org/pdf/2610.37393)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3.1 质量条件策略、§3.2 迭代 hindsight relabeling、Algorithm 1、§4.1 实验设置

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.37393)
- [项目页](https://sinnnj.github.io/qcp)
