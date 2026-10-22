# Much Ado About Noising: Dispelling the Myths of Generative Robotic Control（arXiv:2512.01809）

> 来源归档（ingest）

- **类型：** paper / behavior cloning / generative control / flow matching / diffusion
- **论文：** <https://arxiv.org/abs/2512.01809>；ICLR 2026 论文页：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/9299ca9ed58731945e934adb5b71728c-Abstract-Conference.html>
- **项目页：** <https://simchowitzlabpublic.github.io/much-ado-about-noising-project/>
- **官方代码：** <https://github.com/simchowitzlabpublic/much-ado-about-noising>
- **数据 / checkpoints：** 项目页链接至 Hugging Face datasets 与 checkpoints
- **作者：** Chaoyi Pan、Giri Anantharaman、Nai-Chieh Huang、Claire Jin、Daniel Pfrommer、Chenyang Yuan、Frank Permenter、Guannan Qu、Nicholas Boffi、Guanya Shi、Max Simchowitz
- **机构：** Carnegie Mellon University、Massachusetts Institute of Technology、Toyota Research Institute
- **发表：** ICLR 2026
- **入库日期：** 2026-10-22
- **一句话说明：** 论文通过受控行为克隆基准检验 flow 与 diffusion 控制策略的优势来源，认为关键在受监督的迭代计算与适当随机性，而不是多模态建模或更强表达能力；两步回归式 Minimal Iterative Policy（MIP）可接近 flow 策略表现。

## 研究问题与主要发现

论文对生成式控制策略（GCP）进行受控评测，试图分离生成式目标、策略网络结构、噪声注入和迭代计算各自带来的影响。项目页报告覆盖 28 个任务、状态 / 图像 / 点云 / 语言输入，并包含预训练 VLA 场景。

1. 对齐网络结构与训练设置后，普通回归策略在多数任务可与 flow 策略相当；flow 的优势主要出现在高精度任务。
2. GCP 的表现提升主要来自训练中被监督的迭代计算，并与适量随机性配合。
3. 两步回归式 MIP 在多个基准上基本匹配 flow 策略，说明无需完整流模型采样也可获得迭代计算收益。
4. 随机性与迭代监督有助于策略在分布外观测下贴近专家数据流形，减少落入错误动作模式的情况。

## 开源状态核查（2026-10-22）

官方项目页链接到代码、文档、数据与 checkpoints；代码仓库 README 提供 Robomimic、Kitchen 与 PushT 训练入口，并标注 MIT 许可。结论：**代码、数据与 checkpoints 均有公开入口**。README 提醒，整理后的代码与依赖迁移可能无法逐数复现论文结果；原始结果使用 cleandiffuser Robomimic 环境与数据。

## 对 wiki 的映射

升格为 [wiki/entities/paper-much-ado-about-noising.md](../../wiki/entities/paper-much-ado-about-noising.md)，并与 [离散动作何时能成为好的连续控制器](../../wiki/entities/paper-discrete-actions-continuous-control.md) 及生成式策略 / flow matching 页面互链。

## 一手资料

- 项目页：<https://simchowitzlabpublic.github.io/much-ado-about-noising-project/>
- 论文：<https://arxiv.org/abs/2512.01809>
- ICLR 2026：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/9299ca9ed58731945e934adb5b71728c-Abstract-Conference.html>
- 官方代码：<https://github.com/simchowitzlabpublic/much-ado-about-noising>
