# robomimic 官方项目与研究页面

- **项目主页：** https://robomimic.github.io/
- **研究页面：** https://robomimic.github.io/study/
- **官方代码：** https://github.com/ARISE-Initiative/robomimic
- **官方文档：** https://robomimic.github.io/docs/
- **主论文：** [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://arxiv.org/abs/2108.03298)；CoRL 2022 正式论文页面：[PMLR](https://proceedings.mlr.press/v164/mandlekar22a.html)
- **作者：** Ajay Mandlekar、Danfei Xu、Josiah Wong、Soroush Nasiriany、Chen Wang、Rohun Kulkarni、Li Fei-Fei、Silvio Savarese、Yuke Zhu、Roberto Martín-Martín
- **项目归属：** ARISE Initiative；研究团队来自 Stanford 与 UT Austin 等机构
- **核查日期：** 2026-10-17
- **沉淀到 wiki：** [robomimic 实体页](../../wiki/entities/robomimic.md)

## 官方页面内容摘录

- **项目定位：** 机器人示范学习框架；提供操作任务演示数据集和离线学习算法，目标是让研究者能够复现并公平比较策略与数据集。
- **研究范围：** 论文比较六种离线学习算法，在五个仿真和三个真实世界多阶段操作任务上研究示范质量、观测空间、数据规模与策略选择。
- **研究页公开的主要经验：** BC-RNN、HBC 等使用历史信息的策略在人工示范上表现突出；离线 RL 对混合质量的人类数据较难处理；验证损失与真实策略表现可能不一致；相机观测和超参数选择会显著影响结果。
- **代码与数据：** 项目主页链接官方代码、文档和研究页。v0.1 数据文档提供仿真与真实机器人操作数据的下载说明；Hugging Face 托管仿真数据，真实机器人数据另有下载入口。

## 开放状态核查

- **已开源：** 官方 GitHub 仓库提供可运行训练实现、配置、示例和 MIT 许可。
- **数据：** 官方文档提供数据下载脚本与直接链接；仿真数据可在 [Hugging Face 数据集仓库](https://huggingface.co/datasets/robomimic/robomimic_datasets)获取，真实机器人数据不在该 HF 仓库中。
- **边界：** robomimic 是离线示范学习研究框架，并非特定人形机器人的即插即用部署栈。跨机器人使用需自行适配观测、动作空间、环境封装和控制接口。

## 互指

- [代码仓库归档](../repos/robomimic.md)
- [论文来源归档](../papers/robomimic_arxiv_2108_03298.md)
