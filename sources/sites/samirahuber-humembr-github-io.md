# HUMEMBR 项目页

> 来源归档（site / project-page）

- **标题：** HUMEMBR: Learning Human Routines for Predictive Embodied Navigation
- **类型：** site / project-page
- **官方入口：** <https://samirahuber.github.io/humembr/>
- **论文：** [arXiv:2606.30404](https://arxiv.org/abs/2606.30404) · IROS 2026（页眉声明）
- **代码：** <https://github.com/samirahuber/humembr>
- **机构展示：** Kiel University；George Mason University
- **入库日期：** 2026-08-06
- **一句话说明：** HUMEMBR 官方项目页：摘要、演示视频、BibTeX，以及指向 GitHub **Code** 的入口；论文另称提供代码、视频、基准问题与执行日志。

## 页面公开信息（检索自 2026-08-06）

| 资源 | URL / 状态 |
|------|------------|
| 项目页 | <https://samirahuber.github.io/humembr/> |
| arXiv | <https://arxiv.org/abs/2606.30404> |
| GitHub Code | <https://github.com/samirahuber/humembr> |
| 演示视频 | 项目页内嵌（browser video） |
| 数据集下载 | **未在项目页公开链出**；仓库 README 写明 COBD archive **private** |

## 开源核查（步骤 2.5）

- 项目页导航含 **Code → GitHub**；论文 Abstract 声明 code / videos / benchmark questions / execution logs 见项目页。
- **截至 2026-08-06** 核查：GitHub 含 `src/humembr/`（robot / server / agent / processing / db）、`eval/`、README Quick-start（PostgreSQL+pgvector、GraphNav、KPR 权重、caption 服务、三进程启动）。
- **数据集：** README「Load the released dataset」标明 **currently private**。
- 判定：**代码已开源（可运行部署栈）**；**COBD / PersonEQA 完整数据部分不公开** → wiki 工程实践与局限中写清边界。

## 对 wiki 的映射

- [`wiki/entities/paper-humembr.md`](../../wiki/entities/paper-humembr.md)
- [`sources/papers/humembr_arxiv_2606_30404.md`](../papers/humembr_arxiv_2606_30404.md)
- [`sources/repos/humembr.md`](../repos/humembr.md)
