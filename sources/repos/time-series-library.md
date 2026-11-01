# Time-Series-Library

- **类型：** repo
- **来源：** 清华大学 THUML
- **链接：** <https://github.com/thuml/Time-Series-Library>
- **核查版本：** `4e938a1767106324dd753b2a44832bf870a0252e`（2026-04-18）
- **许可：** MIT
- **数据：** [官方数据集镜像](https://huggingface.co/datasets/thuml/Time-Series-Library)
- **入库日期：** 2026-10-30
- **沉淀到 wiki：** [Time-Series-Library](../../wiki/entities/time-series-library.md)

官方 README 介绍长短期预测、补全、异常检测及分类五类任务，并提供数据准备、依赖安装与实验脚本。源码开放；入口为 `run.py`，实验层为 `exp/`，数据入口为 `data_provider/data_factory.py`，模型位于 `models/`。核查了 README 全文和以上入口、实验基类与长预测实验代码。

维护者于 2026 年 4 月说明不再积极添加新功能，并建议使用较新的 benchmark；基线实现仍可参考。README 的安装方式为克隆仓库、安装 PyTorch 与 requirements，部分模型有额外依赖，不能据此认定库本身可直接 pip 安装。
