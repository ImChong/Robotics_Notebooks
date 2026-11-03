# InstantTrack

> 来源归档

- **类型：** repo
- **链接：** <https://github.com/Huy-Quang-Dao/instantTrack>
- **项目页：** <https://huy-quang-dao.github.io/instantTrack/>（[归档](../sites/instanttrack.md)）
- **许可证：** 代码 MIT；动作数据和衍生文件另受各自许可约束
- **入库日期：** 2026-10-29
- **沉淀到 wiki：** [InstantTrack](../../wiki/entities/instanttrack.md)

## 复现入口和边界

README 提供 `uv sync`、`tools/download_data.sh`、`scripts/train.sh`、`tools/eval_batch.py` 和 `tools/eval_metrics.py`。动作与 G1 资产从 Hugging Face 的 LAFAN1 重定向数据集下载。作者报告使用 RTX 5090、4096 环境、mjlab 与 MuJoCo Warp；实验仅在仿真中进行。SuperMimic 基线是社区复现的 mjlab 移植，不是原论文官方代码。训练时间不包含离线评估。

## 2026-11-03 更新

核查 [InstantTrack 019ae20](https://github.com/Huy-Quang-Dao/instantTrack/tree/019ae20a83c443a057736789b891a77dcc95505e) 的 README、训练入口和主循环。v1.2 增加 Muon、CUDA Graph、批量目标编码和较小策略；4090 同机对照见 [BlinkMimic 归档](./blinkmimic.md)。现有项目页已转为介绍 InstantGMT，原 InstantTrack 页面留在代码库的 `instanttrack-page` 分支。源码子模块尚未完整核验，未运行训练。
