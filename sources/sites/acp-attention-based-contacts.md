# ACP 项目页

- **标题：** Humanoid Mobility via Attention-based Contacts
- **类型：** project site
- **项目页：** <https://leggedrobotics.github.io/ACP/>
- **论文：** [arXiv:2610.31608](https://arxiv.org/abs/2610.31608)
- **代码 / 数据：** 截至 2026-10-23，ACP 项目页未列接触规划器、跟踪器或数据的下载入口。页面说明复用 AME-2 神经地图；AME-2 官方最小代码仓库已公开，但其 README 表明神经地图模型未包含。详见[开放状态核查](../repos/acp-code-availability.md)与[AME-2 官方仓库归档](../repos/ame2-minimal-leggedrobotics.md)。
- **入库日期：** 2026-10-23
- **一句话说明：** 项目页报告以 13.2 分钟人体动作热启动的接触规划与全身跟踪：50 Hz 双网络在 1.5 秒时域规划九个部位的接触，并输出 23 关节动作。
- **感知：** 复用 AME-2 神经地图，Livox MID-360；Jetson Orin NX CPU 上以 10 Hz 处理约 20,000 点扫描，3 cm 网格更新小于 30 ms。
- **技能展示：** 爬梯、攀台、翻越、跳下、起身、楼梯、踩石、窄梁平衡、组合地形与恢复。
- **沉淀到 wiki：** [ACP：注意力式接触规划人形移动](../../wiki/entities/paper-acp-humanoid-mobility.md)

## 资料关系

- 方法与论文摘要摘录见 [arXiv:2610.31608](../papers/acp_arxiv_2610_31608.md)。
- 项目页当前可读取；没有列出 ACP 接触规划器、跟踪器或数据的下载链接。其 AME-2 前作链接指向已公开的官方仓库，但 README 说明神经地图未包含，见[代码状态核查](../repos/acp-code-availability.md)。
