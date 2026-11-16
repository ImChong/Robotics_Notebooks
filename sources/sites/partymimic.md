# PartyMimic — 官方项目主页

> 官方项目技术资料归档（不是转载全文）

- **标题：** PartyMimic: General Humanoid Motion Tracking in Minutes on One GPU（PartyMimic：单卡分钟级通用人形全身运动跟踪）
- **类型：** site / 官方项目主页与完整实验、配方、SuperMimic 消融说明
- **作者 / 团队：** RoboParty Lab
- **项目：** https://roboparty.github.io/PartyMimic/
- **代码：** https://github.com/Roboparty/PartyMimic
- **日期：** 2026-11-11（资料核查日；不推断首次发布日期）
- **核对范围：** 官方项目页全文（Overview、Results、Training data、MLP or attention、Sparse keypoints、Sim2sim、LAFAN speed、Recipe、SuperMimic ablation、Use it）；逐项核对评测表、实验条件、消融与局限。
- **通读：** 已完成
- **开源结论：** 已开源；仓库含训练、数据处理/重定向、评测、ONNX 导出代码与 G1/RP1 两份默认策略 checkpoint；原始数据需遵守各自许可。
- **沉淀到 wiki：** [PartyMimic](../../wiki/entities/partymimic.md)
- **源码归档：** [PartyMimic GitHub](../repos/partymimic.md)

## 项目页提供的一手证据

- 训练：同一 10,731 clip / 41 h 多源动作库，单卡 RTX 4090 默认 MLP 12,000 iter 约 26 min，G1/RP1 的峰值显存分别约 22.6/23.4 GB；attention 约 41/42 min，默认需超过 24 GB。
- 数据：LAFAN1、BONES-SEED 多样性采样、跪爬动作、KungfuAthlete；训练与测试按来源区分，Extreme 按源视频隔离；五组未见动作测试在 MuJoCo sim2sim 中进行。
- G1 held-out：MLP 五组平均身体位置误差 45.1 mm，Mani 20.5 mm；attention 平均 43.5 mm；RP1 MLP 47.8 mm，Mani 16.5 mm（attention Mani 16.1 mm）。
- 稀疏输入：`sparse5` 是骨盆参考加五个关键身体部位（头、双手腕、双脚踝）的姿态，**并非只有五个姿态**；共有六个姿态，按项目页说明的原始输入编码包含 51 个数。
- 复现条件：同卡重训的对照与官方已有 checkpoint 混合列示，不能把所有基线的训练 GPU-hours 当作同条件对比；上述准确度是项目方报告，未独立复测。
- 限制：快动作的单目参考存在滑脚，且高速摆动有执行器极限；最快 Extreme 组平均误差更高。LAFAN-40 的 3–5 分钟针对训练集内回报/零失败阈值，不等同于 41 小时泛化训练。
- SuperMimic 消融：多步闭环梯度与在线重拟合世界模型关键；单步模型梯度、过早冻结世界模型分别导致接近零的回报；实用配方包含 Muon、32 步展开、EMA、参考动作 skip 等。

## 核查范围边界

本归档依据公开项目页面及其结果表，未运行训练或真机验证；不将官方测量误写成第三方复现结果。

## 链接

- [项目主页](https://roboparty.github.io/PartyMimic/)
- [官方 GitHub](https://github.com/Roboparty/PartyMimic)
- [仓库 README](https://github.com/Roboparty/PartyMimic/blob/main/README.md)
- [wiki 实体](../../wiki/entities/partymimic.md)

