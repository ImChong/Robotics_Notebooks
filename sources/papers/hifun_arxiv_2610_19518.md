# HiFun: a Hierarchical Framework for Efficient Functional Dexterous Manipulation Learning（arXiv:2610.19518）

> 来源归档（ingest）

- **标题：** HiFun: a Hierarchical Framework for Efficient Functional Dexterous Manipulation Learning
- **类型：** paper / dexterous-manipulation / hierarchical-rl / human-in-the-loop / dmp / real-world-rl
- **arXiv：** <https://arxiv.org/abs/2610.19518>（PDF：<https://arxiv.org/pdf/2610.19518>；HTML：<https://arxiv.org/html/2610.19518v1>）
- **作者：** Linyi Huang, Guowei Huai, Weibin Liu, Shulong Jiang, Ping Tan, Weixuan Zhang, Hui Zhang, Jie Song
- **机构：** 未核实（arXiv 摘要页未列作者单位）
- **会议：** 项目页标注为 Accepted to CoRL 2026
- **项目页：** <https://hly-123.github.io/HiFun/>
- **代码（项目页链接）：** <https://github.com/Hly-123/HiFun-code>
- **开源核查（2026-10-15）：** **待发布**。项目页标注代码 "Coming soon"；仓库存在但仅含 `assets/` 与 README，README 写明 "Code is coming soon"，无 LICENSE，无训练或运行脚本。项目页未列数据集或权重。
- **文章来源：** [公众号综述摘录](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（文中写作"接触感知手部技能 + 机械臂运动与技能触发"）

## 核心主张（摘要要点）

- 灵巧手功能性操作依赖精确接触、臂手协同与错误恢复；接触动力学难以仿真，需要真机学习。
- 分层：功能手部技能负责精确接触，臂手协调策略控制手臂并决定何时激活手部技能。
- 人工实时接管用于恢复学习；运动基元（DMP）增广降低真机示范成本。
- 6 个真机任务、2 种灵巧手，在线训练约 1 小时内平均成功率 98.3%，对动态扰动具有鲁棒性。

## 平台与数据

- 灵巧手：Sharpa 手（Power-Drill Actuation、Angle-Spreader Retrieval、Thin-Handle Retrieval、Power-Drill Bit Removal）与 XHand（Pipette Dispensing、Thin-Shaft Tool Retrieval；另有 Power-Drill Operation、Faucet Operation 等附加任务）。
- 训练全部在真机上在线进行，未描述仿真训练。
- 示范：单条参考示范经 Cartesian DMP 拟合，随机化初始位姿生成约 100 条自动 rollout；人工接管用 Gello（臂）与按键触发手部技能（手）。

## 方法与结果要点（来自论文 HTML 正文）

- 功能手部技能 π_skill：从运动学参考手势出发，残差策略输出有界的手指位姿修正，利用指尖力反馈调整接触；SAC 真机训练后冻结。
- 协调策略 π_coord：输出 6 维手臂末端增量与技能激活值 α∈[0,1]（缩放冻结手部技能向中性位姿）；在线用 RLPD，离线预训练为 SAC 加全局行为克隆项。
- 多头 critic：导航 Q 头（手臂对齐）与调制 Q 头（激活时机）各自使用 TD 目标。
- 接管优势加权（IAW）：按 critic 估计的优势缩放每次纠正的模仿损失。
- 训练时间（平均）：DMP rollout 收集 18.8 min；在线人在回路微调 46.2 min（上限 1 小时）；BC/DP 的遥操作示范采集 95.0 min。
- 成功率（%）HiFun / HIL-SERL（Glove+Gello）/ HIL-SERL（Ours Interv.）/ BC / DP：平均 98.3 / 4.5 / 13.7 / 23.7 / 41.7。BC 与 DP 每任务使用 200 条成功遥操作示范；每任务 50 次试验。

## 链接

- arXiv 摘要：<https://arxiv.org/abs/2610.19518>
- 项目页：<https://hly-123.github.io/HiFun/>
- wiki 节点：[paper-hifun](../../wiki/entities/paper-hifun.md)
