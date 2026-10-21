# RoboHarn-Evo: Evolving Hierarchical Physical Knowledge for Self-Improving Robotic Manipulation（arXiv:2609.37583）

> 来源归档（ingest）

- **标题：** RoboHarn-Evo: Evolving Hierarchical Physical Knowledge for Self-Improving Robotic Manipulation
- **类型：** paper
- **来源：** arXiv v2；官方项目页与代码仓交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2609.37583>（v1：2026-09-29；v2：2026-09-30）
  - PDF：<https://arxiv.org/pdf/2609.37583>
  - 项目页：<https://ruckbreasoning.github.io/RoboHarn-Evo/>
  - 官方代码：<https://github.com/RUCKBReasoning/RoboHarn-Evo>
- **作者：** Shifeng Bao、Fanding Huang、Yihan Lin、Youhe Feng、Guanlin Li、Chen Zhao、Yang Li、Jiawei He、Cheng Chi、Jing Zhang
- **机构：** 中国人民大学（Renmin University of China）、清华大学（Tsinghua University）、XYZ Embodied AI
- **入库日期：** 2026-10-16
- **一句话说明：** 通过机器人交互反馈持续修订分层物理知识（HPK），在不更新基础模型和低层执行器的前提下，让视觉语言模型驱动的操作系统跨回合复用任务级规划经验与动作级物理经验。

## 核心论文摘录（MVP）

### 1) 问题：长程推理仍受局部物理结果约束

- **链接：** <https://arxiv.org/abs/2609.37583> §1–§2
- **摘录要点：** VLM 可以组织长时程操作任务，但计划是否有效取决于局部交互是否造成预期物理效果。每次执行中的物理反馈不仅用于当前回合，也可成为后续回合可复用的知识。
- **对 wiki 的映射：**
  - [RoboHarn-Evo 实体页](../../wiki/entities/paper-roboharn-evo.md)
  - [VLA 方法页](../../wiki/methods/vla.md)
  - [机器人操作任务页](../../wiki/tasks/manipulation.md)

### 2) 方法：双层知识与双循环

- **链接：** arXiv §3–§4；项目页 Method
- **摘录要点：** HPK 将知识分为 Task Knowledge（子任务选择、顺序、目标与完成条件）和 Action Knowledge（物体相对几何策略与物理效果）。内层执行循环按决策层级检索知识并结合当前场景落地动作；外层更新循环根据轨迹与物理结果修订历史知识、更新适用范围并整理可复用条目。动作效果评估与子任务完成判断分开处理。
- **对 wiki 的映射：**
  - [RoboHarn-Evo 实体页](../../wiki/entities/paper-roboharn-evo.md)
  - [VLA 方法页](../../wiki/methods/vla.md)

### 3) 实验：任务成功、知识纠错与迁移

- **链接：** arXiv 摘要与实验章节；官方 README Results
- **摘录要点：** RMBench 六任务平均中，GPT-5.5 加 HPK 为 82.5%（无 HPK 58.3%，+24.2 个百分点），GPT-6 为 91.7%（72.5%，+19.2 个百分点），Qwen3.8-27B 为 41.7%（19.2%，+22.5 个百分点）。另一项 80 次交互 rollout 实验中，held-out 成功率由 48.3% 升至 75.0%（GPT-5.5）、由 70.0% 升至 88.3%（GPT-6）。作者还报告修复/停用超过 83% 的历史错误知识、保留 95.8% 的有效知识，并从 RMBench 向 RoboDojo 零样本迁移，GPT-5.5 / GPT-6 分别增加 35 / 25 个百分点。
- **对 wiki 的映射：**
  - [RoboHarn-Evo 实体页](../../wiki/entities/paper-roboharn-evo.md)
  - [机器人操作任务页](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射（汇总）

- [wiki/entities/paper-roboharn-evo.md](../../wiki/entities/paper-roboharn-evo.md) — 同一项目的唯一主实体
- [wiki/methods/vla.md](../../wiki/methods/vla.md) — VLA 之外的可演化知识记忆层
- [wiki/tasks/manipulation.md](../../wiki/tasks/manipulation.md) — 长时程操作与真实物理反馈

## 关联资料

- 项目页：[sources/sites/roboharn-evo-project.md](../sites/roboharn-evo-project.md)
- 代码仓：[sources/repos/roboharn-evo.md](../repos/roboharn-evo.md)

## 评测口径提醒

- RMBench 主结果、80-rollout held-out 学习曲线、知识纠错统计、RoboDojo zero-shot transfer 与真机实验是不同实验；引用时需保留各自设置。
- 代码仓提供可运行入口，但复现还需按 benchmark 准备依赖、模型服务、权重/数据与许可资产。
