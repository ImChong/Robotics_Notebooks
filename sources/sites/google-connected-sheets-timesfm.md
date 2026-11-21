# Google Connected Sheets × TimesFM 官方使用文档

> 官方产品文档归档

- **标题：** Use BigQuery ML in Connected Sheets / Forecast data in Connected Sheets using BigQuery ML and TimesFM
- **类型：** site（Google Sheets / BigQuery ML 官方文档）
- **机构：** Google Workspace / Google Cloud
- **产品更新：** 2026-02-17
- **官方帮助：** <https://support.google.com/docs/answer/16923039>
- **官方公告：** <https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html>
- **入库日期：** 2026-11-21
- **核对范围：** 通读官方产品公告、Connected Sheets 帮助中的接入前提、Forecast 创建流程、置信水平与输出列说明；未在个人 Google Cloud 项目实操计费与权限设置。
- **通读：** 已完成

## 使用边界与步骤

1. 数据需来自 **BigQuery 连接的 Connected Sheet**，并可能需要 Google Cloud / BigQuery 权限，非任意离线 Sheets 表格直接零配置运行。
2. Sheets 通过 **Advanced analytics → Create a forecast** 调用 BigQuery ML 里的 TimesFM；无需用户编写 SQL 或训练模型。
3. 选择时间序列与目标数值列、聚合粒度、预测 horizon；可按区域或商品做 breakout。
4. 在输出选项中启用 **prediction levels**，选择置信水平（默认 95%）后，Sheets 结果可显示上下界列。
5. 3.0 Python API 的 0.1/0.9 分位数与 Sheets/BigQuery 给出的可配置 prediction interval **不是同一个默认口径**，分析时要分开标注。
6. Google Workspace 公告称该能力面向 Workspace 与个人 Google 账户推出；具体 GCP 权限和计费仍取决于实际 BigQuery 环境。

## 对 wiki 的映射

- [TimesFM 3.0 项目实体](../../wiki/entities/timesfm-3.md)
