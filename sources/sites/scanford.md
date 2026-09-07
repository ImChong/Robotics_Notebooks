# Scanford（项目页）

> 来源归档（ingest 关联资料）

- **标题：** Scanford — Robot-Powered Data Flywheel Instantiation
- **类型：** site / project-page
- **项目页：** <https://scanford-robot.github.io/>
- **论文：** <https://arxiv.org/abs/2511.19647>
- **机构：** 斯坦福大学（Stanford）；丰田研究所（Toyota Research Institute）
- **入库日期：** 2026-09-07
- **一句话说明：** RPDF 框架的图书馆盘点实例——移动操作机器人在斯坦福东亚图书馆两周野外部署，边扫书架边为 VLM 自动产标注数据。

## 开源核查（步骤 2.5，2026-09-07）

| 链接 | 状态 |
|------|------|
| 项目页 / arXiv | **已发布**（视频、图表、prompt 示例） |
| GitHub / 数据集 / 权重 | **截至入库日未列** |

**结论：** **确认未开源**；可读方法与部署叙事，不能按官方仓复现。

## 页面要点

- **框架图：** RPDF 连接互联网预训练数据与野外杂乱部署之间的鸿沟。
- **任务：** 东亚图书馆书架盘点；中/日/韩书籍、破损标签、遮挡与光照变化。
- **结果摘要：** 2103 书架；18.7 h 人力节省；书识别与困难 OCR 显著提升（见论文 Table/Fig）。
- **Prompt 公开：** 域内书脊标注与域邻接 OCR 的 Gemini/Qwen 评测 prompt 在页内展示。

## 对 wiki 的映射

- [`wiki/entities/scanford.md`](../../wiki/entities/scanford.md)
- [`wiki/entities/paper-scanford-robot-powered-data-flywheel.md`](../../wiki/entities/paper-scanford-robot-powered-data-flywheel.md)
- [`sources/papers/scanford_robot_powered_data_flywheel_arxiv_2511_19647.md`](../papers/scanford_robot_powered_data_flywheel_arxiv_2511_19647.md)
