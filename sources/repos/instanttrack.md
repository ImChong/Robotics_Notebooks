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
