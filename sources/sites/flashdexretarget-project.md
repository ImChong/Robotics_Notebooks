# FlashDexRetarget 项目页

- **URL：** <https://holiday-robot.github.io/FlashDexRetarget/>
- **论文：** <https://arxiv.org/abs/2610.01849>；[PDF](https://arxiv.org/pdf/2610.01849)；[HTML v2](https://arxiv.org/html/2610.01849v2)
- **官方仓库：** <https://github.com/DAVIAN-Robotics/FlashDexRetarget> — 当前托管项目网站源码；算法代码待发布，见[仓库归档](../repos/flashdexretarget.md)
- **补充视频：** <https://holiday-robot.github.io/FlashDexRetarget/static/videos/flashdexretarget_supp.mp4>
- **项目实体：** [FlashDexRetarget](../../wiki/entities/paper-flashdexretarget.md)
- **论文摘录：** [arXiv 来源归档](../papers/flashdexretarget_arxiv_2610_01849.md)
- **作者与机构：** Kyungmin Lee、Sibeen Kim、Dongyoon Hwang、Yoonsang Oh、Donghu Kim、Youngdo Lee、I Made Aswin Nahrendra、Jaegul Choo、Hojoon Lee；KAIST AI、Holiday Robotics
- **入库日期：** 2026-10-07
- **开放状态（2026-10-07）：** 项目页展示论文与演示，Code 按钮为不可点击的 “Code (coming soon)”；当前 GitHub 仓库是项目页源码仓库，尚无训练/推理算法实现、权重或项目数据下载入口。论文评测使用 TACO、OakInk2、HOT3D 数据集，需按各数据集自身许可和申请流程获取。

## 项目简介

FlashDexRetarget 用单个 reference-conditioned RL 策略联合重定向多段人手—物体演示。官方项目页汇总：XHand 50 动作基准 90% 成功率；该实验用 29 GPU-hours，对比 CHORD 的 2,847 GPU-hours，并列出 44 个百分点成功率提升。ArXiv v2 表 I 同时报告严格 SR_MT 手部跟踪成功率为 86%，需与物体级 SR_SPIDER 区分。

项目页当前 README 与页面均说明算法代码将后续发布在上述 GitHub 仓库；现阶段公开仓库主要是项目页文件。尚不能通过该仓库复现论文的训练和评测。

