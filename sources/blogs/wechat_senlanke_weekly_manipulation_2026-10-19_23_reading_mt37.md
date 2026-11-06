# senlanke 周报 10-19–10-23 Manipulation 37 篇 通读

> 来源归档（原文通读 · 支撑 `wiki/overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md`）

- **标题：** 【10.19-10.23前沿论文动态】Manipulation 37篇
- **作者：** senlanke / 具身运控lab（微信公众号）
- **类型：** blog / weekly-digest
- **原文：** <https://mp.weixin.qq.com/s/kn-3vRhkVME86YZupnuSmQ>（2026-11 用 `scripts/fetch_wechat_article.py` 取得全文，约 1.4 万字符）
- **一句话说明：** 按 10 个分栏列出 37 篇 Manipulation 论文，每条含英文标题、中文标题、机构、论文链接、「解决的问题」「创新点」。
- **核对范围：** 通读公众号全文 10 个分栏、37 个条目的「解决的问题 / 创新点」及文末；未逐篇打开各 arXiv 原文（技术事实以各详情节点引用的 arXiv 原文为准），机构多标注「待核实」、未核实。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **分栏与篇数：** VLA / 通用操作策略 7；WAM / World Model 9；在线/离线 RL 与后训练 3；模仿学习与生成式动作策略 3；双臂 2；灵巧 2；柔性/可变形 2；人形上半身 3；长程任务、推理与规划 3；数据集、预训练、跨本体与硬件 3；合计 37。
- **VLA 栏主线：** PhysWord 以原子动作词汇为语义锚点、自回归预测未来 Physical Words（周报称 UniVTAC 90.3%、LIBERO-Long 99.6%）；FocusVLA 按未来意图动态路由多相机视角；Echo 对动作与任务状态信号耦合去噪；T-WLA 联合低频视觉与高频触觉；PACT-VLA 在提交动作块前做推测式验证；Progress Probes P² 为运行时进度监测；FlowVLA 面向倾倒的物理完成状态。
- **WAM 栏主线：** AW-0 的 agentic 世界动作模型；Ropedia W0 用 10,000 小时人类第一视角与 3,000 小时机器人数据，相机位姿表示移动、投影骨架表示操作；UniFlowWAM（语义 + 4D 几何）、UniWALM（语言 + 世界 + 动作）、CW-WAM（接触力旋量）、未来视频表示自蒸馏、Physical Canvas（视觉触觉）、骨架约束潜在动作、AIM（提升动作可控性的数据采集）。
- **RL / 后训练栏：** AtomicRL（原子级 RL 改善长程信用分配）、SimPT（仿真中的干预式 VLA 后训练）、AWLS（优势加权潜在引导，保留单步生成效率）。
- **生成式策略栏：** Critic 加权残差 Flow（视觉插入）、腹腔镜 state-to-vision 蒸馏的分布匹配、Flow Matching 策略中欧拉积分步数的实证研究。
- **双臂 / 灵巧 / 可变形 / 人形栏：** DONIC、基于 3D 重建的双臂预训练；AdaDex、触觉感知手部表示；上下文学习衣物世界模型、实时线状物体重建；COWBOY、WholeBodyMem、SP-HLM。
- **长程 / 数据栏：** DISCO（Progressive-Widening MCTS 在仿真生成长程计划，周期蒸馏为 Policy-Value Model，部署时无需在线仿真搜索）、Self-Evolving Harness、FastHarness；多视角潜在动作预训练、LocoManiLab 测试平台、OpenTac 开放三维触觉生态。
- **局限（原文自身）：** 周报仅给「问题 / 创新点」摘要，不含实验对比；多数条目机构为「待核实」；部分条目标注 CoRL 2026 接收（FocusVLA、AIM、AWLS、One-Step Euler FM）。

## 对 wiki 的映射

- [`wiki/overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md`](../../wiki/overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)
- 逐条目条目索引见 [wechat_senlanke_weekly_manipulation_2026-10-19_23.md](wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
