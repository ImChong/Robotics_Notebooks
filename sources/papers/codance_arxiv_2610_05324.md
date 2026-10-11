# CoDance（arXiv:2610.05324）

> 论文来源归档

- **标题：** CoDance: Learning Reactive and Compliant Human-Humanoid Interaction from Video
- **类型：** arXiv 预印本，v1
- **论文：** <https://arxiv.org/abs/2610.05324>
- **PDF：** <https://arxiv.org/pdf/2610.05324>
- **作者：** Zhuoqun Chen、Shucheng Jia、Boyuan Chen
- **单位：** General Robotics Lab, Duke University
- **提交日期：** 2026-10-04
- **项目页：** <https://generalroboticslab.com/CoDance>（归档：[codance-generalroboticslab.md](../sites/codance-generalroboticslab.md)）
- **代码：** <https://github.com/generalroboticslab/codance>（归档：[codance.md](../repos/codance.md)）
- **数据：** <https://huggingface.co/datasets/GeneralRoboticsLab/CoDance>
- **Wiki：** [CoDance 实体页](../../wiki/entities/paper-codance.md)
- **一句话说明：** 将单段双人舞视频转成机器人参考与模拟舞伴，通过多连杆双手顺应增强训练能跟随且会对外力让步的人形策略。

## 核心论文摘录（MVP）

### 1) 单段视频产生机器人与舞伴参考

论文以两位人类舞者的单段视频为起点，恢复运动后将其重定向成 G1 动作参考与移动舞伴。任务选择双人舞，要求机器人同步脚步并维持连续双手接触。

**对 wiki 的映射：** 输入数据与动作重定向是可规模化来源，但单视频任务先验限制了泛化范围；见实体页「方法栈」「局限与风险」。

### 2) 多连杆顺应增强将运动学演示转成受力数据

作者在双手施加结构化外力，适配多连杆参考轨迹，从纯运动学示范构造 force-aware 训练片段。策略因此学习在互动力作用下如何让步，而不只追踪自由参考。

**对 wiki 的映射：** 增强参考是本文的技术核心；约 80% 腕位移重现率是对训练增强目标的顺应指标，不应误读成任意接触泛化率。

### 3) 仿真训练与实机验证

论文报告仿真策略会根据舞伴变化调整脚步、已见片段的平均顺应位移误差约 2.6 cm，并在 Unitree G1 上实现持续双手接触舞蹈与前后运动转换。

**对 wiki 的映射：** 仿真跟随指标和实机展示回答不同问题；实机证据覆盖窄舞蹈设置，不能代替多任务、多舞伴评测。

## 来源边界

- 本档案按 arXiv v1 论文和官方仓库 README 归纳；若后续出现新版本，应记录版本变化并复核数字。
- 论文是预印本；本归档不将 arXiv 发布等同于同行评审录用。
- 细节分析与工程边界沉淀在 [CoDance 实体页](../../wiki/entities/paper-codance.md)。
