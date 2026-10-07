# Skild AI 技术路线官方资料核对（2026-10-07）

- **类型：** 官方博客 / 论文 / 项目页核查
- **核查日期：** 2026-10-07
- **公司与博客：** <https://www.skild.ai/> · <https://www.skild.ai/blogs>
- **范围：** 公司成立、Skild Brain、视觉运动控制、LocoFormer、人视频学习、工业部署、S1、自博弈；复用已有实体。

## 日期与身份依据

| 事件日期 | 资料与链接 | 核查结论 |
| --- | --- | --- |
| 2023（月份未确认） | [2024-07-09 公司公告](https://www.skild.ai/blogs/announcing-our-300m-series-a) | 官方明确 2023 年成立；2024-07-09 是走出隐身及融资公告日，不能当成立日。 |
| 2025-07-29 | [Building the general-purpose robotic brain](https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain) | Skild Brain 技术介绍：低频高层操作/导航策略与高频低层控制策略；文章回顾的 2024 年结果不表示博客发布于 2024 年。未公开具体 Hz。 |
| 2025-08-06 | [One Model, Any Scenario](https://www.skild.ai/blogs/one-policy-all-scenarios) | Skild Brain 低层视觉运动能力展示：图像和本体感知直接到电机命令；不是另一个有独立版本号的产品。 |
| 2025-09-24 / 09-28 | [Omni-bodied 博客](https://www.skild.ai/blogs/omni-bodied) · [arXiv](https://arxiv.org/abs/2509.23745) · [论文 HTML](https://arxiv.org/html/2509.23745v1) · [项目页](https://generalist-locomotion.github.io/) | 博客为 09-24，论文 v1 提交为 09-28。LocoFormer 作者为 Min Liu、Deepak Pathak、Ananye Agarwal，论文署名机构为 Skild AI。项目页反链公司博客；Light Origins 是后续引用者。 |
| 2026-01-12 | [Learning by watching human videos](https://www.skild.ai/blogs/learning-by-watching) | 当时方案以人视频及少量机器人数据（官方称不足 1 小时）微调；不能把后续 S1 的不改权重 ICL 倒推到本阶段。 |
| 2026-03-19 | [Reindustrial Revolution](https://www.skild.ai/blogs/reindustrial-revolution) | ABB、UR、MiR 合作及 NVIDIA/Foxconn 双臂装配展示；是部署/合作事件，非新模型发布或所有客户已量产完成。 |
| 2026-08-18 | [S1 博客](https://www.skild.ai/blogs/s1) · [官方博客列表](https://www.skild.ai/blogs) | 正文引用格式确认 August 2026，列表明确 Aug 18, 2026；按列表显示日期记录，不虚构统一发布日期。Fig. 8 中 2026-02 初次域内 ICL、2026-05 首次翻煎饼为内部研发回顾，不是独立公开项目首发。 |
| 2026-09-10 | [The Hidden Pillar of Robotics](https://www.skild.ai/blogs/skild-crosses-100m-arr) | 部署数据飞轮与 S1 应用跟进；其中“two weeks ago”是相对叙述，不用于重算 S1 发布日。商业数字为公司自报，本路线不作财务验证。 |
| 2026-09-23 | [Physical Self-Play](https://www.skild.ai/blogs/physical-self-play) | S1-class 模型的自博弈后训练结果，明确在 Skild Brain 框架的预训练/ICL 后；未披露它与 S1 操作演示是否同一份 checkpoint。 |

## 开源核查

截至 2026-10-07，上述公司技术博客及 LocoFormer 官方项目页未提供可下载的官方训练/推理代码、模型权重或数据集入口。项目页只有论文、arXiv、视频和公司博客等链接；因此写为“未见公开资产入口”，不由一个 GitHub 组织的仓库数证明全公司开放状态，也不据此承诺“待发布”。社区实现不能当官方实现。

## 对 wiki 的映射

- [Skild AI 公司与路线](../../wiki/entities/skild-ai.md)
- [LocoFormer](../../wiki/entities/paper-locoformer.md)
- [S1](../../wiki/entities/skild-s1.md)
- [Physical Self-Play](../../wiki/entities/skild-physical-self-play.md)
