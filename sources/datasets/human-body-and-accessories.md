# Human Body And Accessories — Ultralytics Platform

> 来源归档（ingest 配套数据集）

- **标题：** Human Body And Accessories
- **类型：** dataset / human parsing / instance segmentation
- **平台用户：** Muhammadrizwanmunawar（Muhammad Rizwan Munawar）
- **数据集页：** <https://platform.ultralytics.com/muhammadrizwanmunawar/datasets/human-body-and-accessories>
- **旧版/镜像页：** <https://platform.ultralytics.com/platform/datasets/muhammad-human-body-and-accessories>
- **配套模型页：** <https://platform.ultralytics.com/muhammadrizwanmunawar/coco-cihp-train/yolo26n-seg>
- **数据集 URI：** `ul://muhammadrizwanmunawar/datasets/human-body-and-accessories`
- **平台文档：** <https://docs.ultralytics.com/platform/data/datasets>
- **代码框架：** <https://github.com/ultralytics/ultralytics>（通用 Ultralytics 训练框架，不是该数据集专属仓库）
- **入库日期：** 2026-10-05
- **一句话说明：** Ultralytics Platform 上的 19 类人体部件/服饰分割数据集，截图与平台公开索引显示 33,141 张图像、826,524 个标注。

---

## 数据概况（入库快照）

| 指标 | 数值 / 说明 |
|------|-------------|
| 图像总量 | 33,141 |
| 标注总量 | 826,524（平台显示 annotations；不要理解为像素数） |
| 训练 / 验证划分 | 28,142 / 4,999（截图信息） |
| 任务 | Segment（平台标记 Segment Ready） |
| 类别数 | 19 |
| 标签体系 | 细分人体区域、衣物与配件；发布介绍称与 LIP / CIHP 人体解析类别对齐 |
| 数据许可 | 本次可见的平台条目/索引没有足够信息核实许可；使用前须查数据集当前许可与平台访问条件 |

## 截图中可辨认的类别

Face、Upper clothes、Hair、Torso-skin、Left arm、Right arm、Pants、Coat、Right shoe、Left shoe、Left leg、Right leg、Hat、Dress、Socks、Glove、Scarf、Skirt、Sunglasses。

> 类别名称按截图记录，大小写与拼写可能随平台导出版本变化；训练前应以实际导出的 dataset metadata 为准。

## 官方训练入口

配套的 YOLO26n-seg 模型页给出该数据集 URI 作为训练数据来源：

```bash
yolo train model=ul://ultralytics/yolo26/yolo26n-seg data=ul://muhammadrizwanmunawar/datasets/human-body-and-accessories task=segment
```

Ultralytics 文档说明，Platform 数据集可用 `ul://username/datasets/slug` 引用，也可从 Platform 导出 NDJSON。CLI 训练需要按文档配置 API key；本次没有验证外部用户是否能直接克隆此数据集。

## 开源与访问核查

- **数据集页面：** 可由公开搜索索引发现；页面状态显示 Segment Ready。数据文件的实际下载权限、许可与可再分发条件未能从当前公开页面核实。
- **代码：** 未发现该数据集专属训练仓库；可复用的是 Ultralytics 通用训练框架。
- **标签来源关系：** 页面介绍称类别与 LIP、CIHP 对齐；这表示标签体系对齐线索，不能据此认定原始图像来自 LIP/CIHP，也不能认定数据集由二者合并。
- **可复现性：** 获取访问权限、确认数据许可和导出具体版本后，才能精确复现实验；模型页展示了一个训练入口，但不是独立的论文基准报告。

## 对 wiki 的映射

- [Human Body And Accessories 数据集](../../wiki/entities/human-body-and-accessories-dataset.md)
- [平台项目页归档](../sites/ultralytics-human-body-and-accessories.md)
- [Ultralytics 通用代码仓归档](../repos/ultralytics.md)
