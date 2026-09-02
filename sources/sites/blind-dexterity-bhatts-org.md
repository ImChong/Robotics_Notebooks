# BlindDexterity（Blind Dexterity 项目页）

- **标题：** Blind Dexterity: Whole-Body Humanoid Manipulation via Pure Proprioception
- **类型：** site / project-page
- **URL：** <https://aditya.bhatts.org/BlindDexterity/>
- **arXiv：** <https://arxiv.org/abs/2608.29487>
- **入库日期：** 2026-09-02
- **配套论文：** [Blind Dexterity（arXiv:2608.29487）](../papers/blind_dexterity_arxiv_2608_29487.md)

## 一句话摘要

TU Darmstadt IAS / DFKI / hessian.AI 提出的 **纯本体感知人形全身操作** 官方站：Unitree G1 在无相机、无 F/T、无触觉条件下完成 **无 IMU 推抗行走、足球停球、滑板登板、手提箱提柄** 四类任务；强调编码器在柔顺 PD 下的「全身触觉通道」与主动搜索式交互感知。

## 公开信息要点（截至 2026-09-02 复核）

- **机构：** Intelligent Autonomous Systems Lab（TU Darmstadt）；German Research Center for AI（DFKI）；hessian.AI；Jan Peters 组（含 Tongji 合作）。作者：Aditya Bhatt、Oleg Kaidanov、Puze Liu、Jan Peters。
- **平台：** Unitree G1；策略输出关节位置目标，经内置 PD 跟踪；50 Hz 控制。
- **训练：** Isaac Sim + RSL-RL PPO；4096 并行环境；域随机化。
- **四类任务：** encoder-only 行走；足球主动定位停球；滑板姿态消歧与登板；手提箱提柄搜索与提起（含可变刚度 VS 变体）。
- **核心发现：** 从零 RL 的 blind 策略优于 DAgger 蒸馏学生；接触后估计误差骤降；`a_{t-1}` 对利用跟踪残差至关重要。
- **代码 / 数据（步骤 2.5）：** 页头 **Code (to be released)**（GitHub 图标 + `release-status` 文案）；**无** 实际仓库 URL → 按 **宣称将开源 / 待发布** 处理。论文写明任务配置将随代码发布。

## 关联

- Wiki：[paper-blind-dexterity](../../wiki/entities/paper-blind-dexterity.md)
- 交叉：[unitree-g1](../../wiki/entities/unitree-g1.md)、[GentleHumanoid](../../wiki/entities/paper-gentlehumanoid.md)
