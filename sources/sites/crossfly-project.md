# CrossFly 项目页

> 来源归档（ingest · 项目页核查）

- **标题：** Benchmarking Interaction-Oriented Aerial Mobility across Heterogeneous Environments
- **项目：** CrossFly
- **类型：** site（无人机异构环境移动与交互基准）
- **项目页：** https://bozhistudying233.github.io/CrossFly/
- **论文：** [arXiv:2610.18381](https://arxiv.org/abs/2610.18381) · [HTML](https://arxiv.org/html/2610.18381v1) · [PDF](https://arxiv.org/pdf/2610.18381)
- **作者：** Zhibo Duan、Hanlin Kong、Tianran Ma、Yuze Wu、Yuxuan Xu、Kenghou Hoi、Changheng Xie、Xin Zhou、Jianzhe Gao、Fei Gao、Wenguan Wang
- **机构：** 浙江大学；Differential Robotics
- **代码状态：** 项目页标注 “Code soon”；页面当前未提供公开代码仓库链接，暂不视为代码已开源。
- **数据集：** [CrossFly（ModelScope）](https://www.modelscope.cn/datasets/bozhi233/CrossFly)
- **核查日期：** 2026-10-20
- **一句话说明：** CrossFly 将无人机长程导航、跨室内外环境转换与到达后的物理交互准备度纳入同一评测框架。

## 项目页要点

- **规模：** 项目页报告 15,263 条轨迹、23 个重建场景、约 300 万帧、160 多小时飞行数据及 120 km 总航程；跨环境轨迹占 51.2%。
- **数据内容：** 同步的第一视角 RGB 观测、6-DoF 飞行状态、语言指令、航点与子任务标注。
- **任务类型：** 室内、室外、室内↔室外长程轨迹；训练使用 18 个场景，并在另外 5 个未见场景评测。
- **构建流程：** 多视角 RGB 与点云重建场景，标注目标/航点/指令；用 Ego-Planner 与 PD 跟踪生成飞行轨迹，通过碰撞及航点检查筛除不可行样本，再生成并人工核验子任务监督。
- **评测结论：** 项目页给出的未见场景跨环境导航成功率中，CrossFly-Agent 为 9.7%；其他列出的基线为 π₀、π₀-FAST、OpenFly-Agent、OpenVLA-OFT 与 GR00T-N1.7。

## 与论文归档的关系

- [CrossFly 论文摘录](../papers/crossfly_arxiv_2610_18381.md)
