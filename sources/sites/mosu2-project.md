# MOSU2 官方项目主页与 GitHub 展示仓归档

- **类型：** 官方项目主页 / 展示仓
- **项目：** MOSU2: Long-range Navigation System with Multi-modal Scene Understanding
- **主页：** https://jinggm.github.io/mosu2/
- **展示仓：** https://github.com/jingGM/mosu2
- **论文（arXiv v1）：** https://arxiv.org/abs/2611.11148
- **项目站点：** 论文 PDF、视频、架构图、定量结果、BibTeX
- **代码：** 未发现机器人导航系统的训练、推理或部署代码；仅站点 \`index.html\` 与 \`assets/\`、\`mosu2.mp4\`、\`mosu2_paper.pdf\`、\`README.md\`、\`LICENSE\` 等公开文件（核查日期 2026-11-17）。
- **核对范围：** 已通读官方站点 HTML 各区块（标题/作者、摘要、Contributions、Method、Real-world Results、Limitations、BibTeX），并核对 GitHub 根目录全部 7 个条目与 README（仅“# mosu2”）及静态站点 index.html。未声称读过展示仓中另存 PDF 或运行导航程序。
- **通读：** 已完成

## 与 arXiv 版本的显式差异

- **作者：** 项目页列 9 人（包括 Michelle Q. Wang Baldonado 与 Steve Cousins）；arXiv:2611.11148v1 列 7 人。入库的论文引文以 arXiv v1 为准，网站作者用于描述当前网页状态，不将两版混成一版。
- **结果：** 项目页写导航成功率 41%→93%，并列停止手势 0%→67%、信号路口 0%→100%；arXiv v1 Table 2 对应五组不加权均值 61.8%→94.2%、停止手势 61%→71%、信号路口 42%→100%。项目页图片与本地 PDF 可能对应不同稿次，未核实版本映射，须分别引用。
- **时序：** 项目页说 5 帧 / 2.5 s，而 arXiv v1 §4.1 写 5 帧、0.2 s 采样、覆盖 1.0 s。Wiki 采用 arXiv v1 的实验设置。
- **代码开放程度：** GitHub 链接是项目展示网站仓库，不能误标为 MOSU2 导航实现开源；截至核查日未找到可执行导航算法入口。

## 对 Wiki 的映射

- [MOSU2 论文实体](../../wiki/entities/paper-mosu2.md)
- [论文一手归档](../papers/mosu2_arxiv_2611_11148.md)
