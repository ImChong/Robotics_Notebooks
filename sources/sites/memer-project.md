# MemER 项目页

> 来源归档（site）

- **标题：** MemER: Scaling Up Memory for Robot Control via Experience Retrieval
- **链接：** <https://jen-pan.github.io/memer/>
- **论文：** <https://arxiv.org/abs/2510.20328>（ICLR 2026）
- **代码：** <https://github.com/memer-policy/memer>（页面 Code 按钮；2026-10-07 可访问）
- **作者 / 机构：** Ajay Sridhar、Jennifer Pan、Satvik Sharma、Chelsea Finn；Stanford University
- **入库日期：** 2026-10-07
- **一句话说明：** 高层策略选择历史关键帧并生成语言子任务，低层 VLA 执行，面向分钟级记忆的真实长程操作。

## 项目页核查

- **代码状态：** 已公开代码链接；复现仍需自行按仓库 README 配环境。
- **数据 / 权重：** 项目页未提供独立公开数据集或权重下载入口；论文使用演示和少量语言标注微调。
- **项目页结果：** 三个真实任务包含 object search、counting 与 dust & replace；强调历史上下文在重试后分布变化时仍可用。

## 关键方法

高层策略提出候选记忆帧；对候选进行聚类与投票、每簇保留代表帧，压缩成可持续追踪的历史。高层再将关键帧和最新帧转成低层可执行的文本子任务。
