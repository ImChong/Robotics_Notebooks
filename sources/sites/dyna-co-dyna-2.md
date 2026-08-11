# Dyna-2 Research Page（dyna.co/dyna-2）

> 来源归档（ingest）

- **标题：** Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models
- **类型：** research site / company technical report page
- **官方入口：** <https://www.dyna.co/dyna-2>
- **公司首页：** <https://www.dyna.co/>
- **论文 / arXiv：** **无**（页面以 Research 长文 + BibTeX 自引）
- **代码：** **未开源**（页内无 GitHub / HF 链；截至 2026-08-11）
- **数据集：** **未公开**
- **机构：** Dyna Robotics（Redwood City, Calif.；公关稿）
- **入库日期：** 2026-08-11
- **一句话说明：** Dyna Robotics 官方 Dyna-2 研究页：百万小时人类视频 WAM、人→机跨具身缩放律、视频共训消融与生产对照。

## 开源状态（项目页核查，2026-08-11）

| 项 | 状态 |
|----|------|
| Research 正文 | 已发布（约 31 min 阅读） |
| Code / Weights | **确认未开源** |
| Dataset | **未公开** |
| 结论 | **闭源产业研究**；可作缩放律与 WAM 选型参照，不可复现训练栈 |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| §1 Introduction | 三个缩放问题；1M h 人视频；无机器人预训练数据 |
| §2 Architecture | MoT + DiT；flow matching；co-training 边际目标 |
| §3 Scaling laws | 1k–1M 梯子；人 held-out；39 任务零样本机；14 任务后训练真机；视频轴消融 |
| §4 Capabilities | WAM vs VLA、现场 pass、语言跟随、一步视频蒸馏 |
| §5–6 Related / Conclusion | 对照 EgoScale / DreamZero / UWM 等；指向 10M h 路线 |

## 对 wiki 的映射

- 博文摘录：[`sources/blogs/dyna_2_million_hour_wam.md`](../blogs/dyna_2_million_hour_wam.md)
- 沉淀 **[`wiki/entities/dyna-2.md`](../../wiki/entities/dyna-2.md)**
- 公司站：[`sources/sites/dyna-co.md`](./dyna-co.md)
