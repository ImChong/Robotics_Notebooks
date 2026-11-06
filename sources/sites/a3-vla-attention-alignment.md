# A³-VLA 项目页 / 代码开放状态核查

> 来源归档（site）。本项目暂无可核实的项目页与仓库，故不另写 sources/repos 文件。

- **核查日期：** 2026-11-06
- **arXiv 摘要页：** <https://arxiv.org/abs/2611.06447>，仅 v1；摘要页与 PDF 内均未出现项目页或 GitHub 链接（PDF 全文及元数据中只有 arXiv / DOI / CC-BY 链接）。
- **论文自述：** NeurIPS Checklist 第 5 条写「We give the URL of our code」，第 13 条写「We offer our code in an anonymous GitHub repository」；但正文没有该 URL。
- **检索：** WebSearch 查询 "A3-VLA Automatic Perception-Guided Attention Alignment ... github" 仅返回 NeurIPS 2026 会议页与无关论文，无仓库；GitHub 搜索 API 当时返回 502，github.com 搜索页 403，**因此「仓库不存在」不能证实，只能说未找到**。HuggingFace Papers 返回「Paper not found」（说明尚无 HF 模型 / 数据集 README 引用该 arXiv 链接）。
- **数据 / 权重 / 许可：** 论文只称 VLABench 与 RoboTwin 数据集公开；真机数据（Songling 机械臂厨房任务，每任务 80 条演示）、SAM2 伪掩码、微调权重均无开放说明。论文 PDF 为 CC BY 4.0（arXiv 元数据）。
- **结论：** 源码运行时序图不适用（无可读源码）。
