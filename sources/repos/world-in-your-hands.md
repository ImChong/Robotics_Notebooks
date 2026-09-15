# World In Your Hands

> 来源归档（repo + 国内具身开源全景）

- **标题：** World In Your Hands
- **类型：** repo / dataset-devkit
- **机构：** 它石智航（TARS Robotics）
- **链接：** https://github.com/tars-robotics/World-In-Your-Hands
- **论文：** https://arxiv.org/abs/2512.24310
- **项目页：** https://wiyh.tars-ai.com/
- **数据集：** https://huggingface.co/datasets/tars-robotics/WIYH
- **分类：** 数据集/Benchmark
- **许可：** CC BY-NC-SA 4.0
- **入库日期：** 2026-09-06（全景）；2026-09-15（论文 ingest 扩充）
- **一句话说明：** WIYH 官方 devkit：`wiyh.py` 数据加载与可视化、`wiyh2lerobot` 转 LeRobot、`wiyh_tutorial.ipynb`；全量 ~1000 h 数据已上 Hugging Face。

## 开源状态

- **已开源**：公开仓库；样例与全量数据、devkit、LeRobot 转换脚本可获取（以 README 与 HF 为准）。
- **待发布**：README TODO 列出 Foundation Model、Human-centric Challenge；硬件设计文件论文承诺开放但仓内未见 CAD。

## 仓库结构（截至 2026-09-15）

| 路径 | 作用 |
|------|------|
| `wiyh.py` | `WIYH` 类：HDF5 结构树、轨迹/投影可视化 |
| `wiyh_tutorial.ipynb` | 入门教程 |
| `wiyh2lerobot/` | `process_h5.py`、`convert.sh` — WIYH → LeRobot |
| `lerobot/` | LeRobot 相关 vendored/示例 |
| `docs/` | 文档资源 |

## 对 wiki 的映射

- [wiki/entities/paper-wiyh.md](../../wiki/entities/paper-wiyh.md) — 论文实体页（主入口）
- [wiki/entities/cn-os-world-in-your-hands.md](../../wiki/entities/cn-os-world-in-your-hands.md) — 国内开源全景节点
