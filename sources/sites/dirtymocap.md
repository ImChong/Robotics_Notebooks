# DirtyMoCap 官方项目页核查

- **类型：** site / paper / motion-capture
- **项目页：** <https://wanglongzju.github.io/DirtyMoCap-Project-Page/>
- **论文：** <https://arxiv.org/abs/2609.19927>（[摘录](../papers/dirtymocap_arxiv_2609_19927.md)）
- **全文：** <https://arxiv.org/html/2609.19927v1>
- **代码仓库：** <https://github.com/WangLongZJU/DirtyMoCap>（[仓库核查](../repos/dirtymocap.md)）
- **机构：** 浙江大学；西湖大学；复旦大学；国防科技大学；南京大学；新加坡国立大学
- **核查日期：** 2026-10-14
- **一句话说明：** DirtyMoCap 论文展示页、重建视频、方法概述与 HKMALA-Motion 数据说明。

## 项目页要点

- 项目页列明 SIGGRAPH Asia 2026，并链接论文与 GitHub 仓库。
- 方法把无序 marker 先映射为固定身份的人体代理锚点，再以滑动窗口跟踪并拟合 SMPL-H 的姿态、平移与体型。
- HKMALA-Motion 使用传统武术原始光学动捕，页面报告 134 段序列、约 180 分钟、22 个风格类别。
- 页面底部 BibTeX 当前仍写 arXiv preprint；此处按页面上方会议标注和 arXiv 预印本并列记录，不将其擅自表述为正式出版卷期。

## 代码与数据开放核查

| 资源 | 截至 2026-10-14 的核查结论 |
|------|---------------------------|
| GitHub | 项目页链接官方仓库，但仓库当前仅见 README/预览展示，README 明确写 “Code will be released here”；尚无可运行代码入口 |
| HKMALA-Motion | 项目页标注 “Data Coming soon”；论文说明已用原始采集数据生成 SMPL-H 重建，不代表重建数据可立即下载 |
| arXiv 许可 | arXiv HTML 标注 CC BY；该论文许可不等同于武术原始动捕或待发布数据集许可 |

## 对 Wiki 的映射

- [DirtyMoCap 论文与项目实体](../../wiki/entities/paper-dirtymocap-robust-motion-capture.md)
- [动作重定向流水线](../../wiki/concepts/motion-retargeting-pipeline.md)

## 2026-10-30 复核

官方项目页仍标注数据待发布；官方 GitHub 仓库仅有 README，仍写明代码将发布，未见可运行实现。本次未能获取论文全文，尚未重新通读；既有实体已包含项目流程图，源码运行时序图仍不适用。
