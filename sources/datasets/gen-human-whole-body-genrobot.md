# Gen-HumanWholeBody（GenRobot）

- **类型：** 数据集
- **机构：** 简智机器人（GenRobot）
- **数据集：** <https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody>
- **官方项目页：** <https://www.genrobot.ai/data/open-dataset>
- **入库日期：** 2026-10-27

## 官方资料摘要

官方开放数据门户将其描述为全身操作数据集，标注 **600 小时**及全身重建 **MPJPE 约 3 cm**。门户提供 Hugging Face 数据集链接。

## Hugging Face 数据卡摘要（2026-10-27 复核）

- **采集与重建：** 由 DAS-Ego 采集，经 GenRobot 专有 Data Foundation Model（DFM）重建；全身 MPJPE 约 3 cm（上身约 2 cm、下身约 3.5 cm）。
- **规模：** 600 小时、15,943 个 episode、26 个领域；摘要段称覆盖 16,976 个真实场景与 1,281 种技能，概览表另列 16,400 个 unique task，口径以数据卡为准。
- **模态与格式：** 6 路原生 RGB 视角，1600 × 1300，30 FPS，mcap 格式；字段含 `origin`、`face_mark`、`vio`、`handtracking`、`annotation_v2`、`body_track`、`bold_mark`、`task`、`sub_task`。
- **许可与访问：** 数据卡 frontmatter 为 `license: other`，正文未给出许可文本；数据集未设访问门槛（`gated: false`）。

## 2026-10-29 核查补充

- Hugging Face 页面显示文件总大小为 **8.79 TB**；该数字是页面当时显示的仓库文件大小，可能随数据更新变化。
- [官方技术文章](https://www.gen-robot.org/blog/whole-body)介绍：通过同步第一视角设备与多台第三视角相机生成监督真值，DFM 利用可见身体信息、时序线索和人体结构先验重建全身运动；约 3 cm 为官方内部评估结果。
- 数据集页面仍标注 `license: other`，不能将其他 GenRobot 数据集的 CC BY-SA 4.0 许可直接套用到本数据集。

## 开放状态与核查边界

数据可直接从 Hugging Face 下载，但许可条款未公开，商用或再分发前需向官方（opendata@genrobot.ai）确认。官方门户未列出本数据集专属训练代码。

## 对 wiki 的映射

- [Gen-HumanWholeBody](../../wiki/entities/gen-human-whole-body-dataset.md)
- [Gen-HumanEgo](../../wiki/entities/gen-human-ego-dataset.md)
