# CAD 1000 Hours（markov-ai/cad-1000-hours）

> 来源归档

- **标题：** CAD 1000 Hours
- **类型：** dataset / huggingface / computer-use / cad
- **链接：** <https://huggingface.co/datasets/markov-ai/cad-1000-hours>
- **发布方：** [Markov AI](https://huggingface.co/markov-ai)（HF 组织 `markov-ai`）
- **入库日期：** 2026-09-07
- **访问：** 公开（`gated: false`）；Hub 未启用 Dataset Viewer（`viewer: false`）
- **许可：** HF 卡片 **未声明** SPDX / 自定义 License 字段；使用前须在 Hub 页核对条款与再分发限制
- **一句话说明：** 约 **1,021.64 小时**、**597** 条专业 CAD/BIM/结构分析工作流录屏 + 同步键鼠事件 + 任务说明/评分 rubric + 输入输出工程文件；面向 **computer-use agent** 训练与评测。

---

## 规模（HF README，入库日快照）

| 指标 | 数值 |
|------|------|
| 录制工作流时长 | **1,021.64** 小时 |
| 工作流数 | **597** |
| 覆盖软件 | **10** 套（AutoCAD、SOLIDWORKS、CATIA、NX、SketchUp、Revit 等） |
| 文件数 | **8,973** |
| 体量 | **256.60 GiB**（Hub 显示约 276 GB） |
| 下载量 | 约 **112k+**（入库日 HF 统计） |

## 软件分布（小时占比）

| 类别 | 软件 | 工作流 | 小时 | 占比 |
|------|------|--------|------|------|
| 通用制图 CAD | AutoCAD | 238 | 501.99 | 49.14% |
| 机械/产品 CAD | SOLIDWORKS, CATIA, Siemens NX | 282 | 305.27 | 29.88% |
| 建筑/BIM/3D | SketchUp, Revit Architecture/Structure | 67 | 203.82 | 19.95% |
| 结构分析 | STAAD.Pro | 6 | 6.54 | 0.64% |
| 可视化渲染 | V-Ray, D5 Render | 4 | 4.02 | 0.39% |

单软件 Top3：**AutoCAD** 501.99 h、**SOLIDWORKS** 209.04 h、**SketchUp** 142.93 h。

## 单工作流目录结构

路径：`<software>/<workflow-id>/`（如 `autocad/<uuid>/`、`solidworks/<uuid>/`）。

| 文件 / 目录 | 说明 |
|-------------|------|
| `clip.mp4` | 30 FPS 屏幕录制 |
| `events.json` | 与录屏同步的时间戳键鼠事件 |
| `frame_events.json` | 帧级时间对齐 |
| `metadata.json` | 工作流级元数据（如 `platform`, `fps`, `event_count`, `total_duration_ms`） |
| `narration.json` | 帧级自然语言屏幕活动描述 |
| `task_desc.json` | 任务指令与期望交付物 |
| `rubrics.json` | 完成质量评分标准 |
| `task_overview.pdf` | 人类可读任务说明 |
| `input_files/` | 任务给定参考/源文件（无嵌套子目录） |
| `output_files/` | 完成的 CAD 工程与导出交付物（如 `.dwg`） |

## 开源核查（步骤 2.5，2026-09-07）

- **已公开数据：** HF 仓库含完整分软件目录与 README；可用 `huggingface-cli download` / `datasets` 拉取。
- **无独立 GitHub 训练代码：** 本条目为 **纯数据集** 发布；未见官方评测脚本仓。
- **同组织相关集：** [markov-ai/cad-environments](https://huggingface.co/datasets/markov-ai/cad-environments)（51 任务 / 99 h，更结构化评测向）；[computer-use-large](https://huggingface.co/datasets/markov-ai) 系更广桌面录屏语料（若存在，选型时勿混淆）。

## 快速用法

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="markov-ai/cad-1000-hours",
    repo_type="dataset",
    local_dir="cad-1000-hours",
)
```

> 全量约 **256+ GiB**；建议按 `<software>/<workflow-id>` 子树增量下载。

## 对 wiki 的映射

- **wiki/entities/cad-1000-hours-dataset.md** — 数据集实体页
- 交叉：[文字生成 CAD](../../wiki/concepts/text-to-cad.md)、[CAD Skills](../../wiki/entities/cad-skills.md)、[CLI-Anything](../../wiki/entities/cli-anything.md)
