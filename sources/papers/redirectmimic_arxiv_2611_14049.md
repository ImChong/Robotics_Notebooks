# RedirectMimic｜人形全身控制中的局部动作内容编辑

- **英文标题：** RedirectMimic: Local Motion Content Redaction for Humanoid Whole-Body Control
- **论文：** https://arxiv.org/abs/2611.14049
- **全文：** https://arxiv.org/html/2611.14049v1
- **版本：** arXiv v1，2026-11-10
- **作者：** Zepeng Wang、Xiaochuan Shi
- **研究机构：** 武汉大学（Wuhan University）
- **官方项目：** https://xiaopeng-whu.github.io/RedirectMimic/
- **入库日期：** 2026-11-12
- **核对范围：** 官方 arXiv v1 全文的问题设定、局部编辑参考构造、PPO 目标与训练采样、AMASS/mjlab 的评估、对比基线、局限；核对官方项目页的资源链接与展示。未执行训练。
- **通读：** 已完成

## 核心目标与输入输出

对预训练模仿控制策略做**局部内容删改**：例如在规定动作片段中抑制快速甩臂或举手，但仍保留下肢移动、主体姿态和未编辑时间段的动作质量。目标不是生成一条全新的全身运动轨迹，也不是直接控制文本编辑器。

## 方法

1. 在原始动作参考中选定需要删改的时间窗口和身体部位，构造平滑过渡后的**编辑参考**，并相应更新正向运动学和速度。
2. PPO actor 仍接收**原始参考**与当前物理状态；**训练奖励**改为对编辑参考进行跟踪，因此策略学会在不增加新推理条件的情况下自动执行局部抑制。
3. 通过对编辑窗口的重采样，提高稀有局部改动在 on-policy rollouts 中的覆盖，避免直接微调时被大量正常帧稀释。
4. 使用 Unitree G1 29 DoF、mjlab 与 **7,509** 段 AMASS 派生动作。两个测试：S1 手臂速度阈值触发片段，S2 手腕高度越过肩部的举臂片段。

## 实验与重要口径

- S2：RedirectMimic **联合成功率 94.2%**；AntiTrack 为 65.0%，ZeroTrack 为 40.8%。编辑目标动作出现率约 **2.7%**；编辑侧动作跟踪误差约 **0.079 m**。
- S1：RedirectMimic 联合成功率约 **87.4%**；ZeroTrack 为 **89.7%**（略优），不可把 S2 的排名推广至所有测试；RedirectMimic 在编辑参考误差方面更优。
- 作者在未编辑动作上的结果总体接近原始策略，说明“局部改动、整体保真”的目标在所测动作族上较好成立。
- 不同模型的追踪误差、窗口成功率和内容出现率是**不同维度**；“内容出现率越低”不能取代稳定性和整体动作完成度。

## 开放状态

官方项目页提供论文、视频与结果展示。所核对页面**未找到可辨认的训练/部署源码仓库入口**；作者个人主页 https://xiaopeng-whu.github.io/ 与具体项目页不同。未做源码运行时序图。

## 对 wiki 的映射

- [RedirectMimic 实体页](../../wiki/entities/paper-redirectmimic.md)
- [人形机器人移动](../../wiki/tasks/humanoid-locomotion.md)
- [模仿学习](../../wiki/methods/imitation-learning.md)

## 参考来源

- [arXiv v1 完整论文](https://arxiv.org/html/2611.14049v1)
- [作者项目主页](https://xiaopeng-whu.github.io/RedirectMimic/)
