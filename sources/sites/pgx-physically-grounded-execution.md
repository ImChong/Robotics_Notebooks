# PGX | Physically Grounded Execution 官方项目页

> 官方站点来源归档（site）

- **标题：** PGX: Physically Grounded Execution for Long-Horizon Humanoid Loco-Manipulation
- **类型：** site / project page
- **URL：** <https://kl-research.github.io/pgx/>
- **论文：** <https://arxiv.org/abs/2611.26596>（v1，2026-11-19）
- **作者 / 机构：** Kai Lu、Chenyang Ma、Ioannis Havoutis；University of Oxford
- **入库日期及核查日期：** 2026-11-22
- **一句话说明：** 用官方方法图、模拟对比表、Unitree G1 实机演示和视频解释「物理谓词门控 + VLA 技能 + WCL 世界系稳定」的长时程执行框架。

- **核对范围：** 官方项目页 Abstract、Method（Timing/Skills/Precision）、Experimental Results（Table II、III 对应结果）、World-frame stabilization、Real-robot experiments、Task demonstrations、BibTeX，以及页面提供的 Paper、Video 与资源入口。
- **通读：** 已完成

## 开放状态（2026-11-22）

| 资源 | 官方页面核查结果 |
|------|------------------|
| 论文 | arXiv:2611.26596v1，已公开 HTML/PDF |
| 方法与实验视频 | 项目页公开 |
| PGX 官方可运行源代码 | **未列代码仓库或运行说明**；不可直接判定可复现 |
| PGX 模型权重 / 数据集 | **未列独立发布入口** |
| 相关外部项目 | 文中使用的 Isaac Lab Arena / GR00T-WBC 有各自官方仓库，不等于 PGX 框架已开源 |

## 内容要点

- **Timing：** 物理谓词触发状态转换、超时重试、逆向恢复，导航与等待交给执行器。
- **Skills：** 支持 flat 或 stage-trained VLA；项目页 Table III 给出保持权重固定的执行消融。
- **Precision：** 物体相对站位、世界坐标基座保持与按钮目标锚定。
- **实验：** 三项模拟任务（箱搬运、杯放置、咖啡）；真实 G1 使用 44 条真实示范并报告分阶段结果，而不是和模拟混用成功率。

## 对 wiki 的映射

- [论文原文归档](../papers/pgx_arxiv_2611_26596.md)
- [PGX 论文实体](../../wiki/entities/paper-pgx-2611-26596.md)
