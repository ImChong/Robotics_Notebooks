---
type: entity
project_id: raft-reasoning-aligned-finetuning
tags:
  - paper
  - reward-model
  - policy-learning
status: complete
updated: 2026-11-04
arxiv: "2611.04368"
project: https://princeton-prism.github.io/RAFT/
code: https://github.com/princeton-prism/RAFT
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/raft_reasoning_aligned_finetuning_arxiv_2611_04368.md
  - ../../sources/repos/raft_reasoning_aligned_finetuning.md
  - ../../sources/sites/raft-reasoning-aligned-finetuning.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "RAFT（arXiv:2611.04368）：先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。"
---

# RAFT（arXiv:2611.04368）

**RAFT**（*RAFT: Reasoning-Aligned Fine-Tuning Improves Robustness of VLA Post-Training*，[arXiv:2611.04368](https://arxiv.org/abs/2611.04368)，[项目页](https://princeton-prism.github.io/RAFT/)，[代码](https://github.com/princeton-prism/RAFT)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「奖励学习与策略后训练」组。

## 一句话定义

**先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RAFT | RAFT: Reasoning-Aligned Fine-Tuning Improves Robustness of VLA Post-Training | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 微调任务成功，也保住分布外能力。
- 开源结论：**部分开源**（步骤 2.5，2026-11-04）。公众号代码 princeton-prism/RAFT 经 git ls-remote 失败，项目页 HTTP 404，不能当作已开源。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04368](https://arxiv.org/abs/2611.04368) |
| **开源** | **部分开源** |
| **要点** | 先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`princeton-prism/RAFT`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as princeton-prism/RAFT
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 RAFT 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**微调任务成功，也保住分布外能力：先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。**

1. 开源边界：**部分开源** — 公众号代码 princeton-prism/RAFT 经 git ls-remote 失败，项目页 HTTP 404，不能当作已开源。
2. 核心机制：先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [raft_reasoning_aligned_finetuning_arxiv_2611_04368.md](../../sources/papers/raft_reasoning_aligned_finetuning_arxiv_2611_04368.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.04368](https://arxiv.org/abs/2611.04368)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04368)
- [项目页](https://princeton-prism.github.io/RAFT/)
- [https://github.com/princeton-prism/RAFT](https://github.com/princeton-prism/RAFT)
