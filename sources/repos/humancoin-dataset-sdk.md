# FlagOpen/HumanCOIN：多频传感数据集 SDK 与可视化（官方源码核对）

- **类型：** 官方仓库原文及代码核对
- **代码：** https://github.com/FlagOpen/HumanCOIN
- **PyPI 包：** humancoin（官方 README 说明 `pip install -U "humancoin>=0.1.1"`）
- **论文：** https://arxiv.org/abs/2611.26434
- **项目页：** https://flagopen.github.io/HumanCOIN/
- **核对日期：** 2026-11-23
- **核对范围：** 官方仓库 README.md、README_zh-CN.md 的发布/安装/下载/quick start；阅读 `src/humancoin/dataset.py`（`create`、`add_frame`、`_auto_timestamp`、`save_episode`、`__getitem__`）、`src/humancoin/index.py`（`record_frame` / `write` / `load_hf_dataset`）、`src/humancoin/parquet.py`（`add` / `save`）和 `examples/01read_simple.py`；核查 `examples/01write_simple.py` 为 README 的运行前置。未运行环境依赖或下载受限真实数据。
- **通读：** 已完成

## 代码实际支持的读写顺序

- `HumanCOINDataset.create(..., features, fps)` 为多频字段建立元数据和 `MasterIndex`。
- `ds.add_frame("task", text)` 进入 `MasterIndex.record_frame(timestamp, task)`，形成主帧；对普通传感器 `ds.add_frame(key, value, timestamp)` 转到 `ParquetFeature.add` 等独立缓存，缺时间戳时由字段频率与计数器推定。
- `ds.save_episode()` 先写 `MasterIndex.write`，再通过各 feature 的 `save` 输出 Parquet、视频与音频，并作对齐检查；视频像素存 MP4，Parquet 保存时间戳。
- `HumanCOINDataset(repo_id, root)` 读取时，`MasterIndex.load_hf_dataset()` 读取 `master_index.parquet`；`ds[i]` 取得主帧 `timestamp` 后按每个 feature 的配置使用最近样本、时间窗口或插值策略返回数据。
- 以上是**文件读写 SDK 的源码运行时序**，并非注视预测/EEG 去噪模型的完整训练调用，也不是论文所称策略提升的实证。

## 官方 README 可运行入口

```bash
pip install -U "humancoin>=0.1.1"
python -m examples.01write_simple
python -m examples.01read_simple
humancoin-visualize-web /path/to/dataset
```

实际真实数据按数据门户生成对应仓库 ID；多数仓库要求申请访问。脚本 demo 的数据是模拟传感器样本，不是完整语料下载。

## 资源分工

- **源码** Apache-2.0：可公开复用数据集 SDK。
- **语料** CC BY-NC 4.0：非商业许可与独立数据集访问限制。
- **原文映射**：[HumanCOIN 论文](../papers/humancoin_human_intent_egocentric_manipulation_arxiv_2611_26434.md) · [官方项目发布页归档](../sites/humancoin-official-project.md) · [独立实体](../../wiki/entities/paper-humancoin-human-intent-egocentric-manipulation.md)。
