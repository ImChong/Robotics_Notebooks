# Towards Physically Grounded Articulated 3D Assets: A Multi-Agent Pipeline and Benchmark

> 来源归档（ingest · arXiv:2610.13947）

- **类型：** paper / articulated 3D assets / multi-agent pipeline / benchmark
- **arXiv：** [abs](https://arxiv.org/abs/2610.13947) · [HTML](https://arxiv.org/html/2610.13947) · [PDF](https://arxiv.org/pdf/2610.13947)
- **版本与日期：** v1，2026-10-09
- **作者：** Xudong Cai、Ruijie Zhu、Haoliang Shang、Ruining Li、Marc Pollefeys、Shangzhe Wu、Andrea Vedaldi、Chuanxia Zheng
- **项目页：** [Artisan](https://physicalvision.github.io/artisan/)
- **代码：** [physicalvision/artisan](https://github.com/physicalvision/artisan)（项目页与论文配套的官方实现）
- **数据集：** [Artisan-Benchmark](https://huggingface.co/datasets/mastercai/Artisan-Benchmark)
- **一句话说明：** 把文本与可选参考图生成可动、可仿真的 3D 物体，重点将碰撞、重力稳定性、质量和材质纳入生成后的物理验证闭环。

## 研究问题与方法

论文将目标从复现物体外观扩展到复现其运动与功能。Artisan 在 Articraft 程序化资产生成路线之上增加多智能体分工和物理仿真反馈：Design、Coding、Review agents 协作生成与检查资产；MuJoCo 仿真反馈用于发现碰撞、稳定性和物理属性问题，并指导修订。

论文提出覆盖四方面的统一 benchmark：空间保真度、物理合理性、下游任务表现和整体质量。评测材料包括 ArtVIP 与 LightWheel 的参考资产、稳定性 prompts，以及带参考质量的 ABO 物体。

## 论文报告的结果

- 在论文评测设置中，Artisan 在空间保真度方面保持竞争力，并报告了较低的碰撞率和较高的静态及关节状态稳定率。
- 论文消融显示，移除物理验证会显著恶化碰撞表现，说明仿真反馈是其面向可用资产的关键环节。
- 论文还报告：把 Artisan 资产加入机器人训练数据后，π0.5 在未见原生资产上的任务成功率有所提高。该结果属于论文所测任务与数据配置，不应泛化为所有机器人任务。
- 具体指标、基线设置和数据划分以论文表格及附录为准。

## 开源状态

截至 2026-10-25，官方 GitHub 仓库公开，包含生成 CLI、viewer 与 benchmark 评估入口；项目页提供交互式样例。Benchmark 数据集也有 Hugging Face 页面。运行成本、模型 API 配置和数据许可需分别以官方仓库、论文及数据集卡为准。

## 对 wiki 的映射

- **实体页：** [Artisan](../../wiki/entities/artisan.md) — 多智能体生成与物理仿真验证闭环。
- 与 [Articraft](../../wiki/entities/articraft.md) 对照：Artisan 在程序化资产生成基础上强调独立 Review agent 与物理模拟反馈。

## 参考来源

- [arXiv 摘要与全文](https://arxiv.org/abs/2610.13947)
- [Artisan 项目主页](../sites/artisan-project.md)
- [官方代码仓库](../repos/physicalvision-artisan.md)
- [Artisan-Benchmark 数据集](https://huggingface.co/datasets/mastercai/Artisan-Benchmark)
