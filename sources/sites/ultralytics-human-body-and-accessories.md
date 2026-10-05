# Human Body And Accessories（Ultralytics Platform）

> 来源归档（项目页/平台条目核查）

- **标题：** Human Body And Accessories Dataset
- **类型：** Ultralytics Platform dataset page
- **项目页：** <https://platform.ultralytics.com/muhammadrizwanmunawar/datasets/human-body-and-accessories>
- **平台用户页：** <https://platform.ultralytics.com/muhammadrizwanmunawar>
- **模型示例：** <https://platform.ultralytics.com/muhammadrizwanmunawar/coco-cihp-train/yolo26n-seg>
- **训练框架：** <https://github.com/ultralytics/ultralytics>
- **官方数据集文档：** <https://docs.ultralytics.com/platform/data/datasets>
- **入库日期：** 2026-10-05
- **一句话说明：** 平台公开索引收录的人体部件分割数据集；页面索引显示 Segment Ready，配套模型入口引用数据集 URI。

## 项目页与开放状态

| 项 | 核查结果 |
|----|----------|
| 数据页 | 公开索引可见，名称与截图相符 |
| 数据任务 | Segment Ready |
| 规模 | 公开索引与用户截图显示 33,141 张图像、826,524 个标注 |
| 代码 | 未找到该数据集专属代码仓；训练使用通用 Ultralytics 框架 |
| 数据访问 | Ultralytics 文档支持以 `ul://` URI 使用 Platform 数据集及 NDJSON 导出；该数据集自身是否对任意账号开放下载，尚未核实 |
| 许可证 | 当前可读的公开索引未显示可确认的许可条款；使用前需查数据页或平台导出元数据 |

## 标签与来源边界

数据发布说明称标签与 LIP / CIHP 人体解析标签对齐。LIP 与 CIHP 的论文适合作为任务与标签背景参考；目前没有证据证明本数据集复用了它们的原始图像、标注或划分。

## 对 wiki 的映射

- [Human Body And Accessories 数据集](../../wiki/entities/human-body-and-accessories-dataset.md)
- [数据集快照归档](../datasets/human-body-and-accessories.md)
- [Ultralytics 代码仓归档](../repos/ultralytics.md)
