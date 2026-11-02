---
type: entity
project_id: ly-gwm-robocasa-human300
tags:
  - paper
  - world-model
  - adaptation
status: complete
updated: 2026-11-01
arxiv: "2610.23274"
code: https://github.com/SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/ly_gwm_robocasa_human300_arxiv_2610_23274.md
  - ../../sources/repos/ly_gwm_robocasa_human300.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "LY-GWM-RoboCasa-Human300（arXiv:2610.23274）：LY-GWM给策略生成的候选动作预测未来图状态，再用奖励打分筛选；报告局部评测的有限改善，历史基线与官方复评来自不同口径，不能直接混比。"
---

# LY-GWM-RoboCasa-Human300（arXiv:2610.23274）

**LY-GWM-RoboCasa-Human300**（*LY-GWM-RoboCasa-Human300: Prediction-Guided Action Selection with a Graph Dynamics Model*，[arXiv:2610.23274](https://arxiv.org/abs/2610.23274)，[代码](https://github.com/SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「世界模型与在线适配」组。

## 一句话定义

**LY-GWM给策略生成的候选动作预测未来图状态，再用奖励打分筛选；报告局部评测的有限改善，历史基线与官方复评来自不同口径，不能直接混比。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LY-GWM-RoboCasa-Human300 | LY-GWM-RoboCasa-Human300: Prediction-Guided Action Selection with a Graph Dynamics Model | 本文方法 / 基准的简称 |

## 为什么重要

- 预测候选动作结果，再选择执行。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.23274](https://arxiv.org/abs/2610.23274) |
| **开源** | **已开源** |
| **要点** | LY-GWM给策略生成的候选动作预测未来图状态，再用奖励打分筛选；报告局部评测的有限改善，历史基线与官方复评来自不同口径，不能直接混比。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 LY-GWM-RoboCasa-Human300 模型
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

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**预测候选动作结果，再选择执行：LY-GWM给策略生成的候选动作预测未来图状态，再用奖励打分筛选；报告局部评测的有限改善，历史基线与官方复评来自不同口径，不能直接混比。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：LY-GWM给策略生成的候选动作预测未来图状态，再用奖励打分筛选；报告局部评测的有限改善，历史基线与官方复评来自不同口径，不能直接混比。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [ly_gwm_robocasa_human300_arxiv_2610_23274.md](../../sources/papers/ly_gwm_robocasa_human300_arxiv_2610_23274.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.23274](https://arxiv.org/abs/2610.23274)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.23274)
- [https://github.com/SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval](https://github.com/SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval)
