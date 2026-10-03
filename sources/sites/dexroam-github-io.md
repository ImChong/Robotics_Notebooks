# DexRoam 项目页

- **项目页：** <https://dexroam.github.io/>
- **关联论文：** [DexRoam arXiv:2609.35761](../papers/dexroam_arxiv_2609_35761.md)
- **机构：** 香港科技大学、北京智源人工智能研究院、北京大学、北京航空航天大学、中国科学院自动化研究所
- **会议：** 项目页标注 CoRL 2026 已录用
- **代码：** [采集与人机对齐](https://github.com/zhourui9813/DexRoam)、[策略训练](https://github.com/zhourui9813/DexRoam-Policy-Training)
- **数据：** [Hugging Face 示例数据集](https://huggingface.co/datasets/zhourui9813/DexRoam_Realworld_Data)

## 概览

DexRoam 将第一视角人类全身示教转换为移动双臂操作策略的监督信号。项目页报告五项真实机器人任务与两种 VLA backbone：加入对齐的人类示教后，GR00T N1.7 平均成功率由 29% 升至 56%，π0.5 由 32% 升至 57%。

## 开源状态

**部分开放（截至 2026-10-04）。** 项目页提供采集 / 对齐代码和策略训练代码。HF 发布单一任务 50 条真机 episode，数据卡注明其用途是管线验证，不是完整训练集或基准。

## 流程概要

1. Quest 3 采集头部、身体关节和手部关键点，ZED Mini 同步采集第一视角双目 RGB。
2. 经具身映射、相对动作语义映射、任务进度时间重采样，将人类动作转换到机器人动作空间。
3. 将对齐的人类演示与机器人示教结合，训练 VLA 策略。
