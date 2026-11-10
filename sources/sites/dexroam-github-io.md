# DexRoam 项目页

- **项目页：** <https://dexroam.github.io/>
- **关联论文：** [arXiv:2609.35761v1](../papers/dexroam_arxiv_2609_35761.md)
- **机构：** 香港科技大学、北京智源人工智能研究院、北京大学、北京航空航天大学、中国科学院自动化研究所
- **会议：** 作者项目页明确标注 CoRL 2026 accepted
- **代码：** [采集与人机对齐](https://github.com/zhourui9813/DexRoam)、[策略训练](https://github.com/zhourui9813/DexRoam-Policy-Training)
- **数据：** [Hugging Face 单任务示例](https://huggingface.co/datasets/zhourui9813/DexRoam_Realworld_Data)
- **官网复核：** 2026-11-11

## 概览与技术流程

DexRoam 将 Quest 3 + 头戴 ZED Mini 采集的第一视角 RGB、人体姿态及双手关键点，经过三阶段人机对齐，和机器人示教一起供 VLA 学习。具身对齐处理人体/机器人结构差异，动作语义对齐用局部相对动作替代绝对姿态，时间对齐按任务进度匹配机器人 episode 长度；目标是保留底盘、躯干、手臂与手指的连续耦合。

## 关键成果（作者项目页与原论文）

五任务平均成功率 GR00T N1.7 **29% → 56%**，π0.5 **32% → 57%**；在特定数据预算任务中可用 25 条机器人示教 + 人类数据达到/接近 50 条纯机器人示教的参考水平。**不是**一条示教一次部署即学会；原论文报告 0 条机器人示教不能成功。

## 开源状态

**部分开放（截至 2026-11-11）。** 官方项目页同时公开上述两个源码仓与 HF 数据链接。采集/重定向、转换及 VLA 训练入口已可查，但 Hugging Face 只公开 Push Chair & Close Laptop 单任务的 50 条机器人 episode，并非论文完整数据集。代码运行的真实入口见 [采集对齐源码核验](../repos/zhourui9813-dexroam.md) 与 [训练代码核验](../repos/zhourui9813-dexroam-policy-training.md)。

## 对 wiki 的映射

- [DexRoam 论文实体与 Mermaid 图](../../wiki/entities/paper-dexroam-mobile-bimanual-manipulation.md)
