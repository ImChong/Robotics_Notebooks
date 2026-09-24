# LeRobotDataset v3.0 官方文档

- **标题：** LeRobotDataset v3.0
- **类型：** site / 官方文档
- **链接：** https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- **代码仓：** https://github.com/huggingface/lerobot（已开源，Apache 2.0）
- **入库日期：** 2026-09-24
- **一句话说明：** LeRobot **v3.0** 机器人学习数据集格式：Parquet 表数据 + 分相机 MP4 视频 + 关系型元数据；支持 Hub 流式 `StreamingLeRobotDataset`、训练期图像增广、`v2.1→v3` 转换与 `finalize()` 推送约束。
- **沉淀到 wiki：** 是 → [`wiki/concepts/lerobot-dataset-v3.md`](../../wiki/concepts/lerobot-dataset-v3.md)

## 开源核查（2026-09-24）

| 项 | 状态 |
|----|------|
| 文档 | **公开可读**（huggingface.co/docs/lerobot） |
| 实现 | **已开源** — `lerobot >= 0.4.0` 稳定版将内置 v3；此前可用 main 分支或指定 commit 安装 |
| Hub 数据集 | **公开** — `LeRobotDataset(repo_id)` / 流式加载 |

## v3 相对 v2.1 的核心变化（官方 What's new）

- **按文件存储**：多 episode 聚合进单个 Parquet/MP4（v2 为 **每 episode 一文件**）。
- **关系型元数据**：episode 边界与 lookup 由 **metadata** 解析，而非文件名。
- **Hub 原生流式**：`StreamingLeRobotDataset` 无需全量下载。
- **更低文件系统压力**：更少、更大文件 → 初始化更快、大规模更少 inode 问题。
- **统一目录布局**：data / videos / meta 路径模板一致。

## 设计三柱（Format design）

1. **Tabular data**：低维高频信号（state、action、timestamp）存 **Apache Parquet**；经 `datasets` 栈 memory-map 或流式读取。
2. **Visual data**：同 episode 帧拼接编码为 **MP4**；按相机分 shard。
3. **Metadata**：JSON/Parquet 描述 schema（feature 名、dtype、shape）、FPS、归一化统计、episode 在共享文件中的 **start/end offset**。

> 规模化原则：多 episode 的行与视频帧 **concat 进大文件**；episode 视图由元数据重建。

## 目录布局（简化）

| 路径 | 作用 |
|------|------|
| `meta/info.json` | 规范 schema、FPS、codebase 版本、定位 data/video shard 的 path 模板 |
| `meta/stats.json` | 全局 mean/std/min/max；`dataset.meta.stats` |
| `meta/tasks.jsonl` | 自然语言 task → 整数 ID（task-conditioned 策略） |
| `meta/episodes/` | 分块 Parquet：每 episode 长度、task、字节/帧 offset |
| `data/` | 帧级 Parquet shard（多 episode  per file） |
| `videos/` | 每相机 MP4 shard（多 episode per file） |

## API 要点（官方示例）

- **`LeRobotDataset(repo_id)`**：Hub 缓存本地；`dataset[i]` 返回 PyTorch tensor 字典键如 `observation.state`、`action`、`observation.images.*`。
- **`delta_timestamps`**：秒级相对当前帧的多帧窗口，如 `"observation.images.front_left": [-0.2, -0.1, 0.0]` → shape `[T,C,H,W]`。
- **`StreamingLeRobotDataset(repo_id)`**：直接从 Hub 迭代；`lerobot-train --dataset.streaming=true`。
- **Bucket**：`repo_type="bucket"` 流式 HF Storage Bucket；训练 CLI `--dataset.repo_type=bucket`。
- **图像增广**：训练加载时 `image_transforms`（`ImageTransforms` / torchvision `v2`）；**录制时不写增广**，保留原图。
- **可视化增广**：`lerobot-imgtransform-viz --repo-id=...`。

## 迁移 v2.1 → v3.0

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<HF_USER/DATASET_ID>
```

- 聚合 `episode-*.parquet` → `file-*.parquet`；`episode-*.mp4` → `file-*.mp4`。
- 更新 `meta/episodes/*` 的 length、task、offset。

## 推送前必须 `finalize()`

v3 增量写 Parquet + 缓冲 metadata；**`dataset.finalize()`** 在 `push_to_hub()` 前：

- flush episode metadata
- 关闭 Parquet writer 写 footer（否则文件损坏、无法加载）

（官方引用 PR #1903。）

## Lance 可选后端（官方节）

`lerobot-lancedb` 提供 `LeRobotLanceDataset` / `LeRobotLanceVideoDataset`，子类 `LeRobotDataset`，面向大规模随机访问 IO。

## 对 wiki 的映射

- [LeRobotDataset v3.0（概念页）](../../wiki/concepts/lerobot-dataset-v3.md)
- [LeRobot（实体）](../../wiki/entities/lerobot.md)
- [LeRobot 仓库归档](../repos/lerobot.md)
- [LeRobot EnvHub 文档](../sites/lerobot-envhub-docs.md)
