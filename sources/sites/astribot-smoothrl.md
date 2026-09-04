# Astribot SmoothRL 项目页（归档）

- **标题：** SmoothRL: Efficient Online RL During Asynchronous Execution
- **类型：** site / project-page
- **URL：** <https://www.astribot.com/research/SmoothRL>（中文入口 301 → 英文页）
- **英文页：** <https://www.astribot.com/en/research/SmoothRL>
- **arXiv：** <https://arxiv.org/abs/2608.29768>
- **HTML：** <https://arxiv.org/html/2608.29768>
- **入库日期：** 2026-09-02（当时 **404**）；**2026-09-04 再核已上线**（Nuxt，`last-modified` 2026-09-04）
- **配套论文：** [SmoothRL（arXiv:2608.29768）](../papers/smoothrl_arxiv_2608_29768.md)

## 一句话摘要

星尘智能（Astribot）的 **异步推理环内在线 RL** 官方页：hero 演示 + 三任务微调前后对比视频 + 交互式成功率曲线；方法与数字对齐 arXiv:2608.29768。截至 2026-09-04 **未列 GitHub / Hugging Face**。

## 开源状态（步骤 2.5，2026-09-04 复核）

- 页内仅链 [arXiv:2608.29768](https://arxiv.org/abs/2608.29768) 与自有 OSS 媒体。
- **无** Code / Weights / Dataset 按钮；Nuxt 包内无 github/huggingface URL。
- **处理：** **确认未开源**（项目页已上线但无可运行实现）。`## 源码运行时序图` 标不适用。

## 公开信息要点（项目页 + arXiv）

- **机构 / 作者：** Astribot Team。论文 Contributions：Guang Gao\*、Yuxuan Nong\*、Baifu Huang；Project Lead：Jianan Wang（research@astribot.com）。
- **平台：** Astribot S1；冻结任务微调 **π₀.₅**；30 Hz 控制、5 Hz 推理、latency budget **n=6**。
- **三大机制（页内 Core Components）：** 执行区梯度更新；完整时序价值建模（committed 进 critic）；轨迹连续性约束。
- **干预：** 绝对（VR 接管）/ 残差（摇杆增量叠策略）；不打断训练环。
- **三任务 250 ep：** 动态抛投 39%→94%；笔帽合盖 8%→83%；纸箱拆封 30%→90%。页内曲线数字与论文 Table 1 一致。
- **页内叙事：** 异步不是工程选项而是大模型落地必然形态；闭环为「交互—采集—异步推理—在线优化」。

## 演示视频索引（OSS）

**CDN 根路径：** `https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/smoothrl/`

页内「微调前后」：左=冻结基策略失败，右=SmoothRL 成功；每任务 3 段，同环境条件。

| 文件 | URL | 约大小 |
|------|-----|--------|
| hero.mp4 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/smoothrl/hero.mp4> | 12.0 MB |
| dynamic-tossing/01–03.mp4 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/smoothrl/dynamic-tossing/01.mp4> 等 | 180–202 KB |
| pen-capping/01–03.mp4 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/smoothrl/pen-capping/01.mp4> 等 | 256–287 KB |
| box-opening/01–03.mp4 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/smoothrl/box-opening/01.mp4> 等 | 210–250 KB |

同名 `.jpg` 为封面（已核 `hero.jpg` 200）。

框架图（站点相对路径 `/images/research/`）：`smoothrl-framework-base.png`、`smoothrl-framework-flows.png`、`smoothrl_absolute_intervention.jpg`、`smoothrl_residual_intervention.jpg`、`smoothrl_tasks.jpg`、`smoothrl_eval_protocol.jpg`。

## 关联

- Wiki：[paper-smoothrl](../../wiki/entities/paper-smoothrl.md)
- 交叉：[ARLI](../../wiki/entities/paper-arli.md)、[Lumo-2](../../wiki/entities/lumo-2.md)、[Philia](../../wiki/entities/philia.md)
