---
type: entity
project_id: beyondcontact
tags:
  - paper
  - tactile
  - dexterous
status: complete
updated: 2026-11-04
arxiv: "2611.03059"
code: https://github.com/iit-DLSLab/BeyondContact
related:
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/dexterous-kinematics.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/beyondcontact_arxiv_2611_03059.md
  - ../../sources/repos/beyondcontact.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "BeyondContact（arXiv:2611.03059）：为Isaac Lab加入实验辨识的气流力模型，结合机械臂夹爪旁的风机训练推吹协作；通过真机零样本部署探索超出直接接触范围的物体操作。"
---

# BeyondContact（arXiv:2611.03059）

**BeyondContact**（*BeyondContact: A Reinforcement Learning Framework for Airflow-Based Non-Contact Manipulation*，[arXiv:2611.03059](https://arxiv.org/abs/2611.03059)，[代码](https://github.com/iit-DLSLab/BeyondContact)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「触觉与灵巧操作」组。

## 一句话定义

**为Isaac Lab加入实验辨识的气流力模型，结合机械臂夹爪旁的风机训练推吹协作；通过真机零样本部署探索超出直接接触范围的物体操作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BeyondContact | BeyondContact: A Reinforcement Learning Framework for Airflow-Based Non-Contact Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 让机械臂学会推，也学会吹。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.03059](https://arxiv.org/abs/2611.03059) |
| **开源** | **已开源** |
| **要点** | 为Isaac Lab加入实验辨识的气流力模型，结合机械臂夹爪旁的风机训练推吹协作；通过真机零样本部署探索超出直接接触范围的物体操作。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`iit-DLSLab/BeyondContact`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as iit-DLSLab/BeyondContact
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 BeyondContact 模型
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

**让机械臂学会推，也学会吹：为Isaac Lab加入实验辨识的气流力模型，结合机械臂夹爪旁的风机训练推吹协作；通过真机零样本部署探索超出直接接触范围的物体操作。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：为Isaac Lab加入实验辨识的气流力模型，结合机械臂夹爪旁的风机训练推吹协作；通过真机零样本部署探索超出直接接触范围的物体操作。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [tactile-sensing](../concepts/tactile-sensing.md)
- [contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [beyondcontact_arxiv_2611_03059.md](../../sources/papers/beyondcontact_arxiv_2611_03059.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.03059](https://arxiv.org/abs/2611.03059)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.03059)
- [https://github.com/iit-DLSLab/BeyondContact](https://github.com/iit-DLSLab/BeyondContact)
