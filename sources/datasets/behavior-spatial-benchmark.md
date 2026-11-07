# BEHAVIOR-Spatial 官方数据集归档

- **官方 HF：** https://huggingface.co/datasets/behavior-spatial-benchmark/BEHAVIOR-Spatial
- **论文：** https://arxiv.org/abs/2610.34051
- **代码：** https://github.com/arielchen07/BEHAVIOR-Spatial-benchmark
- **核查日期：** 2026-11-07
- **许可：** 数据 MIT；不包含仿真 3D assets；BEHAVIOR-1K 资产许可只限非商业学术研究（需独立接受条款）。

## 数据目录与统计（官方数据卡）

| 目录 | 数量与说明 |
|---|---|
| `vqa/general/single_frame` | 1,078 道问答 |
| `vqa/general/multi_frame` | 494 道（旋转 239 + 导航 255） |
| `vqa/task/manipulation` | 1,082 道，14 操作任务 |
| `vqa/task/navigation` | 602 道，10 导航任务 |
| `ir/gt` | 1,240 个目标定位查询；12 任务有掩膜，其余按盒评分 |
| `action/eval/manipulation/held_out` | 469 个 |
| `action/eval/manipulation/prompt_probe` | 700 个 |
| `action/eval/navigation/held_out` | 401 个 |
| `action/train/manipulation` | 5,532 episodes（其中 4,833 unique；另含 699 重复以匹配训练采样） |
| `action/train/navigation` | 3,912 episodes |

- **格式：** 训练轨迹 LeRobot v2.1；23 维动作/状态，30 fps；原生头部视角 720×720、手腕 480×480，论文策略使用缩放到 224×224 的输入。
- **下载边界：** 数据卡展示总量约 183GB，混合 JSON 与视频导致 HF Dataset Viewer 无法直接统一预览；评测时需从原始文件读取。
- **核对范围：** 已核对官方 Hugging Face 数据卡全文与仓库 README 引导；未下载或逐项解析全量文件。
- **通读：** 已完成

## 对 wiki 的映射

- [唯一项目实体](../../wiki/entities/paper-behavior-spatial.md)
- [论文](../papers/behavior_spatial_arxiv_2610_34051.md)
- [官方源码](../repos/behavior-spatial-benchmark.md)
