# Benchmarking Interaction-Oriented Aerial Mobility across Heterogeneous Environments

> 来源归档（ingest · arXiv:2610.18381v1）

- **类型：** paper / aerial robotics / benchmark / aerial mobile manipulation
- **arXiv：** [abs](https://arxiv.org/abs/2610.18381) · [HTML](https://arxiv.org/html/2610.18381v1) · [PDF](https://arxiv.org/pdf/2610.18381)
- **版本与日期：** v1，2026-10-13（cs.RO）
- **作者：** Zhibo Duan、Hanlin Kong、Tianran Ma、Yuze Wu、Yuxuan Xu、Kenghou Hoi、Changheng Xie、Xin Zhou、Jianzhe Gao、Fei Gao、Wenguan Wang
- **机构：** 浙江大学；Differential Robotics
- **项目页：** [CrossFly](https://bozhistudying233.github.io/CrossFly/)
- **数据集：** [CrossFly（ModelScope）](https://www.modelscope.cn/datasets/bozhi233/CrossFly)
- **代码状态：** 项目页标注 “Code soon”，当前未给出代码仓库链接；不应表述为已开源。
- **一句话说明：** CrossFly 评测无人机是否能在室内、室外及二者转换的长程轨迹中完成导航，并在抵达后处于适合下游物理交互的状态。

## 研究问题

现有空中机器人基准通常只覆盖单一环境；涉及交互的测试又常从目标附近开始，因而没有充分评估长程导航与后续交互之间的衔接。论文提出 CrossFly，把异构场景中的移动与交互准备度纳入统一基准。

## 基准与评测设计

- **任务覆盖：** Indoor、Outdoor、Indoor↔Outdoor 三类轨迹，重点评估室内外环境转换下的长程空中移动。
- **数据规模：** 15K 轨迹、20 多个重建场景。项目页进一步报告 15,263 条轨迹、23 个场景、约 3M 帧、160+ 飞行小时和 120 km 总航程；跨环境轨迹占 51.2%。
- **标注与观测：** 项目页列出的数据包括同步第一视角 RGB、6-DoF 飞行状态、语言指令、航点和子任务标注。
- **Interaction Readiness Rate（IRR）：** 衡量导航终态是否适合下游交互，不要求绑定某一种操作策略；它补充导航成功率，便于观察“到达了”是否意味着“可以接着操作”。
- **场景划分：** 论文评测已见与未见场景；项目页说明训练使用 18 个场景，另有 5 个场景用于未见场景测试。
- **对比方法：** 论文评测 5 种代表性空中 VLA agent，并提出分层式 CrossFly-Agent：高层 VLM 规划器指定当前子任务，低层 VLA 输出 6-DoF 飞行动作并预测是否调用操作技能。

## 主要结果

- 未见场景中，跨室内外设置比单一环境更难。项目页表格中 CrossFly-Agent 的导航成功率为室内 17.3%、室外 18.5%、跨环境 9.7%。
- 论文摘要将未见场景最高成功率报告为 9.7%；解读时应连同“跨环境”这一更难设置一并说明，避免误写成所有设置的最高值。
- 下游操作案例研究支持 IRR 与后续物理交互相关，但 IRR 衡量的是交互就绪状态，并不直接代表操作成功。

## 对复现与选型的提示

CrossFly 更适合研究长程空中导航与移动—交互衔接的基准评测；如果只关心目标附近的局部操作，单看该导航成功率不足以判断操作能力。项目页已提供 ModelScope 数据集入口，但截至 2026-10-20 代码仍标注 “Code soon”，复现流水线的代码开放状态需后续重新核查。

## 参考来源

- [arXiv 摘要与版本信息](https://arxiv.org/abs/2610.18381)
- [arXiv HTML 全文](https://arxiv.org/html/2610.18381v1)
- [CrossFly 项目页](../sites/crossfly-project.md)
- [CrossFly 数据集（ModelScope）](https://www.modelscope.cn/datasets/bozhi233/CrossFly)
