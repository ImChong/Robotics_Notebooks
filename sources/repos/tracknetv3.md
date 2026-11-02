# TrackNetV3

- **类型：** repo
- **链接：** https://github.com/qaz812345/TrackNetV3
- **论文：** https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf
- **入库日期：** 2026-11-01
- **开放状态：** 官方仓库包含训练、推理代码和模型权重入口；以仓库具体说明为准。
- **作用：** 利用连续帧与背景预测羽毛球位置热图，再可选用 InpaintNet 修复轨迹。
- **核查范围：** 论文 2023 版；官方 `predict.py` 推理入口。未实际运行。

论文表 1 在 TrackNetV2 数据集上报告 accuracy 97.51%、precision 97.79%、recall 99.33%、F1 98.56%。这是论文测试条件下球定位指标，不是完整战术分析的准确率。

## 待完成

通读论文并核查相关模型、数据处理与评测源码；编写项目实体、项目流程图和源码运行时序图。
