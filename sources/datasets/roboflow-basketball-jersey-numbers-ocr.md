# basketball-jersey-numbers-ocr（Roboflow Universe）

> 来源归档（dataset）

- **标题：** basketball-jersey-numbers-ocr
- **类型：** dataset / ocr / basketball / sports-analytics / player-identification
- **机构：** 罗博福流（Roboflow）
- **链接：** <https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr>
- **关联仓库：** [`sources/repos/roboflow_sports.md`](../repos/roboflow_sports.md) — README「Reading jersey numbers」挑战与 datasets 表
- **入库日期：** 2026-09-29
- **一句话说明：** 篮球 **球衣号码 OCR** 标注集，对应 Roboflow 在体育 CV 中列出的核心难点之一：模糊、背对、遮挡下的号码读取；**数据已挂 Universe**，[`roboflow/sports`](https://github.com/roboflow/sports) 尚无与 `examples/soccer` 同级的端到端 OCR demo 模式。
- **为什么值得保留：** 与 [Roboflow Sports](../../wiki/entities/roboflow-sports.md) 的 **SigLIP 分队**（外观聚类、非号码身份）形成对照——号码 OCR 是统计与再识别链路的下一跳；入库便于 lint 与后续 ingest 跟进 hosted 模型 / 训练 notebook。

## 开源与页面核查（2026-09-29）

| 项 | 结论 |
|----|------|
| 数据入口 | Universe URL 可公开访问；详情页部分环境需浏览器验证（Cloudflare） |
| 与代码仓关系 | README **datasets** 表直接链到本集；**无** `examples/basketball/` 或 `JERSEY_OCR` CLI 模式 |
| 挑战语境 | 同 README：*Reading jersey numbers* — blur、转身、遮挡 |

> **说明：** 入库日未抓取到稳定的图像计数/API 元数据（Universe 需 API key 或完整浏览器会话）。规模与类别以 Universe 项目页为准；后续 lint 可补全。

## 与 roboflow/sports 工程栈的对照

| 能力 | soccer demo（已有） | 篮球 OCR 集（本归档） |
|------|---------------------|------------------------|
| 球员身份 | `TeamClassifier`（SigLIP + UMAP + KMeans，两队） | 号码级监督信号（OCR 标签） |
| 可运行示例 | `examples/soccer/main.py` 六模式 | 数据集 + Universe 训练/部署入口；**待社区 notebook** |
| 检测 backbone | 默认 YOLOv8（AGPL） | 可与 [RF-DETR](../../wiki/entities/rf-detr.md) / Inference SDK 同栈微调 |

## 对 wiki 的映射

- 主实体：[`wiki/entities/roboflow-sports.md`](../../wiki/entities/roboflow-sports.md)
- 球场关键点姊妹集：[`roboflow-basketball-court-detection-2.md`](./roboflow-basketball-court-detection-2.md)
