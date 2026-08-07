# RekaDaily-10k（raw）— Hugging Face

> 来源归档

- **标题：** RekaDaily-10k (raw)
- **类型：** dataset / huggingface
- **链接：** <https://huggingface.co/datasets/RekaAI/RekaDaily-10k-raw>
- **研究发布页：** <https://reka.ai/labs/research/rekadaily-10k-egocentric-household-manipulation-data>
- **机构：** 瑞卡人工智能（Reka AI） / Claru
- **入库日期：** 2026-08-07
- **许可：** Apache 2.0（ungated）
- **访问：** 公开，无需申请门控
- **一句话说明：** RekaDaily-10k 的 **raw 档**：WebDataset tar 打包的无剧本第一人称日常/家务视频，按采集项目分 shard；增量发布，目标全量 **10,312** 小时。

---

## 规模（HF README，入库日快照）

| 指标 | 数值 |
|------|------|
| 当前可用 | 约 **886** 小时 / **39,643** 视频 / **1,877** shards（README 口径） |
| 全量目标 | **10,312** 小时（研究页；「下周初」完成上线叙事，以 HF 实际为准） |
| 格式 | WebDataset `.tar`（约 5 GB/shard）+ parquet 元数据 |
| 模态 | 视频（mp4/mov）+ JSON sidecar；无机器字幕（raw） |

> Hub 文件树可能随增量上架继续增长；工程以 README「Current contents」与 `metadata/*.parquet` 为准，勿只数 tar 文件。

## 目录结构

```text
data/<project>/shard-NNNNN.tar   # <video_id>.{mp4|mov} + <video_id>.json
metadata/browse.parquet          # 每视频一行：160px 缩略图 + 全量元数据
metadata/index.parquet           # 同上，无缩略图（轻量程序读）
metadata/thumbs.parquet          # 320px 静帧（datasets Image 列）
sample/                          # ~400 条散装视频，便于快速浏览
```

### 采集项目（README 列出）

- `egocentric_household_tasks`
- `egocentric_household_tasks_usa`
- `egocentric_commercial_environments`
- `residential_egocentric_latam_upload_via_claru`
- `video_capture_activities`
- `video_capture`
- `video_capture_first_person_videos_phone`

## 元数据字段

| 字段 | 说明 |
|------|------|
| `video_id` | 与媒体文件名一致 |
| `project` | Claru 采集项目 |
| `flow`, `activities` | 活动类项目的会话场景 + 动作 taxonomy |
| `category`, `subcategory` | video-capture 类项目的类别 taxonomy |
| `lighting` | 光照条件（若有） |
| `duration_s`, `fps`, `width`, `height`, `num_frames`, `codec` | probe 统计 |
| `collector` | 加盐哈希采集者 ID（不同值 ≈ 不同环境） |

每条视频只填充一套 taxonomy（`flow/activities` **或** `category/subcategory`）。

## 快速用法

```python
import webdataset as wds

url = (
    "https://huggingface.co/datasets/RekaAI/RekaDaily-10k-raw/"
    "resolve/main/data/egocentric_household_tasks/shard-00000.tar"
)
ds = wds.WebDataset(url)
```

Dataset Viewer 默认打开 `browse` 配置；`metadata` 配置便于无图程序读。Hub 对视频 WebDataset 的预览当时平台侧有问题，但 tar 可正常下载/流式读取。

## 隐私与下架

- 容器元数据（GPS、设备标识、拍摄时间戳）已剥离。
- 另有自动化 PII 筛查（见研究页）。
- 问题反馈：`contact@reka.ai`。

## 对 wiki 的映射

- **wiki/entities/rekadaily-10k-dataset.md** — 数据集实体页
- **sources/sites/rekadaily-10k.md** — 研究发布与两档设计叙事
