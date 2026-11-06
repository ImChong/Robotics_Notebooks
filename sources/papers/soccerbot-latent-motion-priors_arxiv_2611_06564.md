# SoccerBot：Playing Soccer Like Humans Do（arXiv:2611.06564）

> 来源归档（paper）

- **类型：** paper
- **标题：** Playing Soccer Like Humans Do: Learning Versatile and Agile Humanoid Soccer via Latent Motion Priors
- **arXiv：** <https://arxiv.org/abs/2611.06564>（cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.06564>
- **版本：** v1，2026-11-04 14:28:40 UTC（1,768 KB）；2026-11-06 核对时无 v2。草稿登记文件（v1 摘要级）与此版本一致，无版本出入。
- **作者：** Haoyu Shao, Yunshen Wang, Jingze Zhang, Shaohang Zhu, Yong-Lu Li, Tianzhu Zhang, Baoxiong Jia（通讯）, Siyuan Huang（通讯）
- **机构：** 中国科学技术大学；BIGAI 通用人工智能国家重点实验室；上海交通大学；北京大学
- **会议：** CoRL 2026（正文页脚：10th Conference on Robot Learning, Austin TX）
- **项目页：** <https://soccerbot-humanoid.github.io/>
- **入库日期：** 2026-11-06（已通读 14 页全文与附录 A–C）

## 核心摘录（人工转述，数字取自原文）

- 方法三阶段（§3、Fig. 2）：运动跟踪教师 → 蒸馏潜运动先验（后验编码器 + 仅本体的状态条件先验 + 解码器；损失为 MSE + KL + 后验均值平滑，式 (1)）→ 冻结先验与解码器，PPO 训练技能条件残差策略（z = z^p + α z^s，式 (2)；残差边界惩罚式 (3)）。
- 数据：Soccer-X [26]（参考文献 [26] = SMGDiff, ICCV 2025）经 GMR [27] 重定向，仿真回放过滤力矩限制下无法跟踪的片段，保留约 2.5 小时（§3.1）。
- 观测 106 维、critic 109 维（Table A.1）；奖励按技能路由（式 (A.2)，Table A.2）；TSI：p_TSI = 0.3，每技能 256 个终止状态（附录 C.1）；8192 并行环境（附录 C）。
- Table 1：Ours 踢球 SR 97.04、带球 SR 89.88、停球 SR 83.08、移动 SR 100.00；Train ref. MMD² 0.226，Post-train ref. 0.554。
- Table 2：技能切换 SR（含 w/o TSI 对照）；Table 3：踢球 Near/Mid/Incoming/WideYaw 仿真 98.6/87.0/72.9/83.9，真机 95.0/85.0/70.0/80.0。
- Table 4：真机 20 次/技能，踢球 VelErr 47.38%、DirErr 10.41°；带球 VelErr 30.17%、ResT 2.37 s；停球 StopT 1.84 s、DistErr 1.07 m；移动 VelErr 21.53%。
- 球动力学：e = 0.654，滚动减速度 0.100 m/s²（§4.5）。
- 局限（§6）：sim-to-real 差距因技能而异、依赖外部动捕、仅四种技能、仅单机器人。

## 未在论文中给出

- PPO 与蒸馏的具体超参数（学习率、α、β_KL、β_sm、训练时长）；过滤前动捕原始规模；真机试验的场地/地面材质/球型号。

## 对 wiki 的映射

- [paper-soccerbot-latent-motion-priors](../../wiki/entities/paper-soccerbot-latent-motion-priors.md)
