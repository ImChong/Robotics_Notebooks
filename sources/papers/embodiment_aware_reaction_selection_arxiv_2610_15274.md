# Embodiment-Aware Inference-Time Search（arXiv:2610.15274）

> 来源归档（ingest）

- **标题：** Embodiment-Aware Inference-Time Search for Human Reaction Selection in Humanoid Retargeting
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.15274>
- **HTML：** <https://arxiv.org/html/2610.15274v1>
- **作者：** Yongpeng Cao、Masahiro Hirano、Yuji Yamakawa
- **机构：** 东京大学生产技术研究所（Institute of Industrial Science, The University of Tokyo）；东京大学跨学科信息学府（Interfaculty Initiative in Information Studies, Graduate School of Interdisciplinary Information Studies, The University of Tokyo）
- **提交日期：** 2026-10-11
- **代码：** 论文列出匿名入口 <https://anonymous.4open.science/r/Runtime-Humanoid-Reaction-Searching-2BF2>；截至 2026-10-20，入口内容、许可证与可运行步骤尚未独立核验。
- **一句话说明：** 固定人体反应生成器，在推理时采样候选并经过目标人形重定向，以交互保真与本体可行性选择反应。
- **对 wiki 的映射：** [paper-embodiment-aware-reaction-selection.md](../../wiki/entities/paper-embodiment-aware-reaction-selection.md)

## 核心论文摘录

### 1) 反应生成器作为候选提议器
冻结 ReMoGen 检查点；从观测到的人体交互历史推断交互语义，再采样多个因果反应候选。主配置为 K=16，候选 0 是单样本、无搜索基线。

### 2) 先重定向，再按目标本体筛选
每个候选都经冻结重定向器转换为机器人动作。候选需通过人体/机器人动作有效性和语义下限检查，再按机器人运动学可行性与交互保持收益排序；无合格候选时回退到候选 0。

### 3) 评测结果与边界
- Inter-X 官方测试集、GMR + Unitree G1 主设置：可行性增益 +1.61%（95% CI [+1.51, +1.72]），机器人有效率 96.15%→96.73%。
- 人类动作空间与 G1 目标本体的最优候选在 78.98% 样本中不同；排序 Spearman 相关系数为 0.366。
- 还在 NMR、InterHuman、H1-2 上报告泛化结果；G1 与 H1-2 的最优候选有 38.23% 不同。
- RTX 4090 + i9-13900KF 上，完整加载的 K=16 路径中位耗时 108.68 ms（约 9.2 Hz）；论文说明流程结束于选出的 G1 动作，**没有实机驱动**。

## 来源核查

作者单位、方法、数值与代码入口来自论文 arXiv HTML；论文正文说明代码将通过上述匿名 4open.science 地址提供。当前无法由该入口核验源码结构、许可证或命令行运行路径，因此不把代码状态写成已复现。

