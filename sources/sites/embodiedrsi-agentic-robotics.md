# EmbodiedRSI — 作者官方项目主页与公开状态核验

- **项目主页：** https://geeksongs.github.io/agentic_robotics/
- **论文：** https://arxiv.org/abs/2610.10498
- **作者网站源码：** https://github.com/Geeksongs/agentic_robotics （默认 `website` 分支）
- **知识节点：** [EmbodiedRSI](../../wiki/entities/paper-embodiedrsi.md)
- **论文归档：** [arXiv:2610.10498v1](../papers/embodiedrsi_arxiv_2610_10498.md)
- **机构：** 哥伦比亚大学（Columbia University）、普林斯顿大学（Princeton University）
- **核验日期：** 2026-11-08
- **核对范围：** 官方项目主页标题、作者机构、Abstract、Method 三阶段、RoboCasa365 / LIBERO-Pro 数值与图、五组实机图像和结果表、页脚；通过作者 GitHub 仓库 README 核查目录/发布目的，区分网站源码与算法实现。
- **通读：** 已完成

## 页面直接可核实的事实

- Fast System：假设图 + Value-of-Information 选择试验，再联合更新 code / skill。
- Slow System：M1 物理记录、M2 假设证据、M3 抽象认知；记忆动作根据后续 Fast System 收益训练。
- 仿真结果：RoboCasa365 Overall 77.0%、Composite-Unseen 71.3%；LIBERO-Pro Overall 86.8%。
- 真机：SO-101 + frozen SmolVLA，作者报告五任务总成功率 46.0%→71.3%；眼镜桥抓取任务 60.0%→56.7%，**没有逐任务全胜**。
- zero-shot 的严格对象是 **在仿真演化的 harness → 真机**。基础 SmolVLA 已按每任务 50 条遥操作示范先行微调。

## 开源结论（关键）

- **官方算法实现：** 页面未列可运行训练/评测/部署的 code 链接；截至核验日期**未核实公开**。
- **公开的网站源码：** [Geeksongs/agentic_robotics](https://github.com/Geeksongs/agentic_robotics) README 明确 “Project website”；提供 `python3 -m http.server 8000` 静态预览与 `scripts/build_site.py` 网站生成，**不是论文系统运行时序**。README 说明论文原始稿、数据映射和部分素材是本地私有内容、未公开。
- **权重与数据集：** 项目页未列可独立下载的 EmbodiedRSI 权重或数据集入口；不能等同已开源。
- **站点笔误：** 项目页 Abstract 中有 “EmbodiedEvo” 字样，但论文与项目名称、其余版块均为 **EmbodiedRSI**；收录以论文身份为准。
- **附录边界：** 论文 v1 公开 PDF 未含引用的附录 A–I；算法复现/超参数无法完整检查。

## 参考来源

- [官方项目主页](https://geeksongs.github.io/agentic_robotics/)
- [作者网站 README](https://github.com/Geeksongs/agentic_robotics/blob/website/README.md)
- [arXiv 论文 v1 PDF](https://arxiv.org/pdf/2610.10498)
