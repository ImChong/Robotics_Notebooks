# CS2-10k / RekaCS2-10k — Hugging Face

> 来源归档

- **标题：** CS2-10k: A Large-Scale Egocentric Counter-Strike 2 Dataset
- **类型：** dataset / huggingface
- **链接：** <https://huggingface.co/datasets/RekaAI/CS2-10k>
- **新闻页：** <https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset>
- **渲染器：** <https://github.com/reka-ai/cs2-dem-renderer>
- **Viewer：** <https://huggingface.co/spaces/RekaAI/CS2-10k-viewer>
- **机构：** 瑞卡人工智能（Reka AI）
- **入库日期：** 2026-08-07
- **许可：** CC BY-NC 4.0（attribution, non-commercial；底层 demo 版权归原权利人）
- **访问：** 公开，ungated
- **一句话说明：** 职业 CS2 比赛渲染的大规模 egocentric 游戏视频 + 逐帧控制/状态标注；WebDataset tar（约 2 GB/shard），按地图分目录。

---

## 规模与格式

| 指标 | 数值 |
|------|------|
| 视频 | **600,000+** player-round clips |
| 时长 | **10,000+** 小时 |
| 分辨率 / 帧率 | **720p · 48 fps** |
| 打包 | WebDataset：`data/<map>/*.tar`（约 2 GB/shard） |
| 索引 | 顶层 `index.parquet`（含 `shard` 定位） |
| 样本 | `<uuid>.mp4` + `<uuid>.parquet` |

### 地图分片（HF API 快照，入库日）

| map | tar shards（约） |
|-----|------------------|
| ancient | 7965 |
| nuke | 6391 |
| dust2 | 5294 |
| mirage | 4266 |
| overpass | 3976 |
| train | 2706 |
| inferno | 125 |
| **合计** | **~30,723** |

## 标注 schema（clip 级）

| 字段 | 类型 | 说明 |
|------|------|------|
| `map` | string | 地图名 |
| `round_number` | int | 回合号 |
| `team` | int | **0 = Terrorist，1 = Counter-Terrorist**（以 HF README 为准；新闻页曾写反） |
| `num_frames` | int | 帧数 |
| `fps` | float | 48.0 |
| `total_time` | float | 秒 |
| `fov` | float | 90.0° |
| `frame_data` | list[dict] | 逐帧数组 |

### 逐帧字段

| 字段 | 说明 |
|------|------|
| `actions` | 活跃键拼接：W/A/S/D、J 跳、C 蹲、R 走、V 自由落体、`[` 开火、`]` 开镜/副、`-` 无输入 |
| `mouse_x_delta` / `mouse_y_delta` | 水平/垂直视角增量 |
| `position_x/y/z` | 世界坐标（游戏单位） |
| `rotation_yaw` / `rotation_pitch` | 相机朝向 |

## 引用（HF card）

```bibtex
@misc{cs2-10k,
  title  = {CS2-10k: A Large-Scale Egocentric Counter-Strike 2 Dataset},
  author = {Reka AI},
  year   = {2026},
  url    = {https://huggingface.co/datasets/RekaAI/CS2-10k}
}
```

## 对 wiki 的映射

- **wiki/entities/rekacs2-10k-dataset.md** — 数据集实体页
- **sources/sites/rekacs2-10k.md** — 新闻与用例叙事
- **sources/repos/cs2-dem-renderer.md** — 可扩展渲染管线
