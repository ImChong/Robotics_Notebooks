# DexRoam: Learning Mobile Bimanual Dexterous Manipulation from Egocentric Whole-Body Human Demonstrations

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.35761>
- **项目页：** <https://dexroam.github.io/>
- **代码：** <https://github.com/zhourui9813/DexRoam>（采集与对齐）、<https://github.com/zhourui9813/DexRoam-Policy-Training>（策略训练）
- **数据：** <https://huggingface.co/datasets/zhourui9813/DexRoam_Realworld_Data>（示例子集）
- **作者：** Rui Zhou、Yibo Yuan、Junkai Zhao、Fangyuan Zhao、Xiaoguang Zhao、Shanghang Zhang、Sirui Han
- **机构：** 香港科技大学、北京智源人工智能研究院、北京大学、北京航空航天大学、中国科学院自动化研究所
- **年份 / 会议：** 2026 / CoRL 2026（项目页标注已录用）
- **一句话说明：** 用消费级 VR 头显和头戴双目相机捕获全身人类示教，经具身、动作语义和时间对齐后，与机器人示教联合训练移动双臂操作策略。

## 关键贡献

- 通过 Meta Quest 3 与 ZED Mini 捕获第一视角 RGB、人体姿态与手部关键点，无需外部动作捕捉设备。
- 将人类动作迁移拆为具身对齐、动作语义对齐、时间对齐三步。
- 保留行走、躯干、双臂、头部和手指动作耦合，并用 VLA 策略学习。
- 论文报告五项真实机器人任务的平均成功率：GR00T N1.7 从 29% 提升至 56%，π0.5 从 32% 提升至 57%。

## 开源状态

**部分开放，已有可运行代码。** 官方项目页链接到采集 / 人机对齐仓库和策略训练仓库。Hugging Face 发布了 Astribot + 双 XHand 单任务示例（50 条 episode）；数据卡明确说明这是验证加载与训练流程的示例，不是完整训练集或论文基准。

## 对 wiki 的映射

- 论文实体页：[paper-dexroam-mobile-bimanual-manipulation.md](../../wiki/entities/paper-dexroam-mobile-bimanual-manipulation.md)
- 任务背景：[双臂操作](../../wiki/tasks/bimanual-manipulation.md)、[移动操作](../../wiki/tasks/loco-manipulation.md)
- 项目归档：[DexRoam 项目页](../sites/dexroam-github-io.md)
- 代码归档：[采集与对齐](../repos/zhourui9813-dexroam.md)、[策略训练](../repos/zhourui9813-dexroam-policy-training.md)
