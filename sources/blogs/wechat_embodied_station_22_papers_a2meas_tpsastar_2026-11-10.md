# 从失败诊断到动作纠正，机器人还缺怎样的接口？（具身智能小站 · 22 篇）

> 来源归档（blog / 微信公众号）

- **标题：** 从失败诊断到动作纠正，机器人还缺怎样的接口？
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Ugm4meP1cQm-R1omnGtpyQ
- **发表日期：** 2026-11-10
- **入库日期：** 2026-11-10
- **抓取方式：** `scripts/fetch_wechat_article.py` 取正文
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_22_papers_a2meas_tpsastar_2026-11-10.md`](../raw/wechat_embodied_station_22_papers_a2meas_tpsastar_2026-11-10.md)
- **核对范围：** 文章全文读完（22 条：1 篇深读 A²-MEAS、2 篇跟进 TPS-A* 与被动折叠四旋翼、19 篇扫读；每条为一句话判断加结果说明加论文 / 代码 / 项目页链接）；文末评论未读；文章没有需要看的配图。22 篇论文里 20 篇新建节点，每篇逐篇通读论文并核对小站说法；其余 2 篇（DKWBC、ToolDex）复用既有节点（均已是 3/3），**沿用既有页面，没有重新读论文**。
- **通读：** 已完成
- **一句话说明：** 2026-11-10 更新的 22 篇具身智能新论文，按阅读优先级分三档；每篇论文对应唯一详情节点（新建 20 / 复用 2）。
- **说明：** 小站只给一句话判断和结果说明；技术事实以各详情节点引用的 arXiv 原文为准。

## 22 篇 → 本库节点

| # | 档 | 论文节点 | arXiv | 状态 |
|---|----|----------|-------|------|
| 01 | 深读 | [抓空之后，让纠错真正改变下一步动作](../../wiki/entities/paper-a2-meas-agentic-embodied-intelligence.md) | [2611.09124](https://arxiv.org/abs/2611.09124) | 新建 |
| 02 | 跟进 | [异形机器人过窄门，按真实轮廓检查碰撞](../../wiki/entities/paper-tps-astar.md) | [2611.09210](https://arxiv.org/abs/2611.09210) | 新建 |
| 03 | 跟进 | [让四旋翼接触折臂，穿过比机身窄的缝](../../wiki/entities/paper-soft-morphing-quadrotor-gap-traversal.md) | [2611.09272](https://arxiv.org/abs/2611.09272) | 新建 |
| 04 | 扫读 | [给导航模型的自信分数加一道校准](../../wiki/entities/paper-cpnav.md) | [2611.10366](https://arxiv.org/abs/2611.10366) | 新建 |
| 05 | 扫读 | [从第一视角视频补出接触与相对压力](../../wiki/entities/paper-touchannotator.md) | [2611.09129](https://arxiv.org/abs/2611.09129) | 新建 |
| 06 | 扫读 | [把安全示范与高收益示范分开学](../../wiki/entities/paper-dexdice.md) | [2611.09433](https://arxiv.org/abs/2611.09433) | 新建 |
| 07 | 扫读 | [给物体目标，让双手自己学会交接](../../wiki/entities/paper-simduoreal-object-centric-bimanual.md) | [2611.10410](https://arxiv.org/abs/2611.10410) | 新建 |
| 08 | 扫读 | [自由握笔写字，同时学会稳笔与控力](../../wiki/entities/paper-dexwriting.md) | [2611.10451](https://arxiv.org/abs/2611.10451) | 新建 |
| 09 | 扫读 | [感知更准，也可能错过整场任务](../../wiki/entities/paper-embodiedintentbench.md) | [2611.10498](https://arxiv.org/abs/2611.10498) | 新建 |
| 10 | 扫读 | [跟随指定行人，先保住目标身份](../../wiki/entities/paper-dobottrack.md) | [2611.09041](https://arxiv.org/abs/2611.09041) | 新建 |
| 11 | 扫读 | [学人类示范，先对齐动作发生的阶段](../../wiki/entities/paper-batch-behavior-alignment.md) | [2611.10148](https://arxiv.org/abs/2611.10148) | 新建 |
| 12 | 扫读 | [先筛候选动作，再想象执行后果](../../wiki/entities/paper-tiresias.md) | [2611.10392](https://arxiv.org/abs/2611.10392) | 新建 |
| 13 | 扫读 | [动作跟踪交给动力学，平衡交给残差](../../wiki/entities/paper-dkwbc.md) | [2611.09605](https://arxiv.org/abs/2611.09605) | 复用 |
| 14 | 扫读 | [用可复用交互模板训练人形搬运](../../wiki/entities/paper-sugarv2-humanoid-interaction-reference.md) | [2611.09266](https://arxiv.org/abs/2611.09266) | 新建 |
| 15 | 扫读 | [没有标准工具时，记录人如何借物完成任务](../../wiki/entities/paper-tooldex.md) | [2611.09036](https://arxiv.org/abs/2611.09036) | 复用 |
| 16 | 扫读 | [把人体运动知识迁移到人形惯性估计](../../wiki/entities/paper-unified-inertial-encoder.md) | [2611.10137](https://arxiv.org/abs/2611.10137) | 新建 |
| 17 | 扫读 | [互动投入度评测，把儿童与成人一起纳入](../../wiki/entities/paper-multimediate-26-engagement-estimation.md) | [2611.09913](https://arxiv.org/abs/2611.09913) | 新建 |
| 18 | 扫读 | [在路径跟踪外加一层轻量避障过滤](../../wiki/entities/paper-lacart.md) | [2611.09640](https://arxiv.org/abs/2611.09640) | 新建 |
| 19 | 扫读 | [搬椅子前，先选有利于最终摆放的抓点](../../wiki/entities/paper-grasp-tags.md) | [2611.09864](https://arxiv.org/abs/2611.09864) | 新建 |
| 20 | 扫读 | [用四部手机记录柔性物体的受控运动](../../wiki/entities/paper-lagsplat-mpov-dataset.md) | [2611.09959](https://arxiv.org/abs/2611.09959) | 新建 |
| 21 | 扫读 | [世界模型上机运行，先看数据搬运的代价](../../wiki/entities/paper-wmpa-world-model-planning-accelerator.md) | [2611.10316](https://arxiv.org/abs/2611.10316) | 新建 |
| 22 | 扫读 | [让社交机器人按情境说好第一句话](../../wiki/entities/paper-proactive-conversation-opener-cco.md) | [2611.09906](https://arxiv.org/abs/2611.09906) | 新建 |

## 对 wiki 的映射

- [具身智能小站 11-10：22 篇论文阅读坐标](../../wiki/overview/embodied-station-2026-11-10-22-papers-technology-map.md)
