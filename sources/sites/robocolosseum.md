# RoboColosseum 项目官网与排行榜

> 来源归档（site）

- **标题：** RoboColosseum — The Proving Ground for Physical AI
- **类型：** site
- **链接：** https://robocolosseum.ai/#top
- **论文：** https://arxiv.org/abs/2611.02497
- **代码：** https://github.com/RoboColosseum/RoboColosseum
- **数据集：** https://huggingface.co/RoboColosseum
- **排行榜：** https://robocolosseum.ai/leaderboard
- **接入文档：** https://huggingface.co/docs/lerobot/main/en/robocolosseum
- **入库日期：** 2026-11-04；2026-11-21 复核
- **核对范围：** 阅读官网首页的评测赛道、提交/连接/运行/打分流程、硬件 FAQ、结果数据公开政策与论文引用；核对官方 GitHub 汇总仓与 LeRobot 的 RoboColosseum 集成说明。动态排行榜不固化排名。
- **通读：** 已完成
- **沉淀到 wiki：** [paper-robocolosseum](../../wiki/entities/paper-robocolosseum.md)

## 官方产品定义和两个赛道

官网强调在真实机器人上开展多站点评测，而非仿真排行榜；描述现阶段覆盖 140+ scenes（官网实时宣传口径，**不是**论文 2026-10-28 结束时 124 个站点的实验统计）。

- **Open：** 同一场地/指令匿名 A/B 直接部署对比，人工评估赢家或平局。
- **Fine-tune：** 预先确定的任务；各模型从统一的 100 条示范微调，每个任务做 25 次尝试。
- **提交方式：** 网站申请 → 选择本体 → 自托管推理服务器通过 Policy API 接入，或提交 checkpoint 由现场机器托管 → 现场操作员执行和评分 → 展示榜单。
- **开源与数据：** 官方索引 GitHub 包括 robot client、policy server 和本体特定 Fine-tune 工具；每次试验（含失败）视频和数据按 LeRobot 格式发布到 Hugging Face。不能将其描述为全运营后台开源。
- **硬件：** SO-101、Franka/DROID、Bimanual YAM、Unitree G1；Earth Rover Mini+ 官方标记正在加入。

## 注意区分

此项目是 **RoboColosseum**（双 s，真机分布式物理智能评测）。[RoboColiseum](../../wiki/entities/robocoliseum.md)（单 s，智元 Genie Sim）是完全不同的仿真竞技场。原始 2026-11-04 的“未列 GitHub”历史判断已于本次源码核查中更新。