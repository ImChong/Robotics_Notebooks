# touch-scale.github.io（TouchScale 项目页）

> 来源归档（ingest · 项目主页）

- **标题：** TouchScale: 500 Hours of Human Vision and Touch for Visual-Tactile Learning
- **类型：** site / project-page
- **官方入口：** <https://touch-scale.github.io/>
- **配套论文：** <https://arxiv.org/abs/2610.10288>
- **官方代码：** <https://github.com/phai-lab/touchscale>
- **数据集：** <https://huggingface.co/datasets/2077AIDataFoundation/TouchScale>
- **核查日期：** 2026-10-19
- **对应 wiki：** [TouchScale](../../wiki/entities/touchscale.md)

## 项目页与开放状态核查

本次直接抓取项目主页失败，站点未能通过网页读取工具打开。项目主页链接由 Hugging Face 数据卡明确列出；其公开内容与开放状态通过数据卡和官方 GitHub 仓库交叉核验：

- **代码：已公开。** 官方仓库包含 TouchScale 质量检查流程及 N0-VTLA 视触觉 mid-training / robot post-training 代码。
- **数据：部分公开。** Hugging Face 数据卡截至核查日发布 100 小时（15,324 episodes）；完整 500 小时仍标注为 2026 年 11 月前计划发布。下载访问需登录并同意分享联系信息。
- **许可：** 数据卡标注 CC-BY-NC-4.0；代码仓库主体为 MIT，但 `n0-vtla/` 目录沿用 CC BY-SA 4.0，模型权重受 Gemma Terms of Use 约束。

## 关联归档

- [论文来源](../papers/touchscale_arxiv_2610_10288.md)
- [数据集发布记录](../datasets/touchscale_huggingface.md)
- [官方代码仓库](../repos/phai-lab-touchscale.md)
