# basketball-court-detection-2（Roboflow Universe）

> 来源归档（dataset）

- **标题：** basketball-court-detection-2
- **类型：** dataset / keypoint-detection / basketball / sports-analytics / broadcast-vision
- **机构：** 罗博福流（Roboflow）
- **链接：** <https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2>
- **关联仓库：** [`sources/repos/roboflow_sports.md`](../repos/roboflow_sports.md) — README「⚽ datasets」表与篮球侧几何挑战
- **入库日期：** 2026-09-29
- **一句话说明：** 广播视角 **篮球场关键点** 标注集：单类 `court` 上的场线/角点式关键点，用于估单应、俯视投影或与足球 `football-field-detection` 同构的训练 Colab 扩展。
- **为什么值得保留：** 补齐 [Roboflow Sports](../../wiki/entities/roboflow-sports.md) 在 **篮球** 侧的可复现数据入口；与 README 列出的「球衣 OCR」集并列，支撑球场标定与高级统计（跑动距离、速度）前的 **相机–场地平面** 对齐。

## 规模与形态（Universe 页面，2026-09-29 核查）

| 项 | 内容 |
|----|------|
| 图像数 | **850**（页面统计） |
| 类别 | **1** — `court`（关键点检测任务） |
| 数据集版本 | 19（页面统计） |
| 公开模型 | 17 个版本（含 RF-DETR Preview Keypoint 等 hosted 模型） |
| 任务类型 | **Keypoint Detection**（Roboflow Universe） |

## 与 roboflow/sports 的关系

- 官方 [`roboflow/sports` README](https://github.com/roboflow/sports#-datasets) 将本集列为 **🏀 basketball court keypoint detection** 下载入口；库代码与 `examples/soccer/` demo 仍以足球为主，篮球侧 **无同级 `examples/basketball/`**，但几何模块（`ViewTransformer`、球场配置模式）可类比扩展。
- 足球对照集：<https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi>

## 引用（Universe 提供的 BibTeX 模板）

```bibtex
@misc{ basketball-court-detection-2_dataset,
  title = { basketball-court-detection-2 Dataset },
  type = { Open Source Dataset },
  author = { Roboflow },
  howpublished = { \url{ https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2 } },
  url = { https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2 },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2026 },
  month = { aug },
  note = { visited on 2026-09-29 },
}
```

## 对 wiki 的映射

- 主实体：[`wiki/entities/roboflow-sports.md`](../../wiki/entities/roboflow-sports.md)
- 足球场线/关键点方法对照：[`wiki/methods/soccer-field-line-detection.md`](../../wiki/methods/soccer-field-line-detection.md)
