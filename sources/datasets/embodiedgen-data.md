# EmbodiedGenData（HuggingFace）

> 来源归档

- **标题：** EmbodiedGenData — Sim-Ready 3D Asset Library
- **类型：** dataset
- **链接：** <https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData>
- **浏览：** [EmbodiedGen-Gallery-Explorer](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer)
- **论文 / 项目：**
  - V2：[arXiv:2607.07459](https://arxiv.org/abs/2607.07459)
  - V1：[arXiv:2506.10600](https://arxiv.org/abs/2506.10600)
- **代码：** <https://github.com/HorizonRobotics/EmbodiedGen>
- **机构：** 地平线（Horizon Robotics）
- **入库日期：** 2026-07-14
- **许可：** Apache-2.0
- **一句话说明：** EmbodiedGen 管线产出的 **跨格式 sim-ready 3D 资产库**（URDF / MJCF / USD / mesh / affordance / 预览视频），以 `dataset_index.csv` 索引约 **4.1K** 条资产，总体量约 **346 GB**。

---

## 规模与索引

| 项 | 数值（ingest 快照） |
|----|---------------------|
| 索引行数 | **~4,168** 条（`dataset/dataset_index.csv`） |
| 总体积 | **~346 GB** |
| HF 规模标签 | `1K<n<10K` |
| 默认 split | `train` → `dataset/dataset_index.csv` |

## `dataset_index.csv` 字段

| 列 | 说明 |
|----|------|
| `uuid` | 资产唯一 ID |
| `primary_category` / `secondary_category` / `category` | 分层类目（如 `bathroom_supplies` → `soap_dish`） |
| `description` | 英文物体描述 |
| `asset_dir` | 相对 `dataset/` 的资产目录 |
| `urdf_path` | URDF 相对路径 |
| `video_path` | 预览视频 |
| `has_gs_ply` / `has_usd` / `has_interaction` | 是否含 3DGS / USD / 交互标注 |
| `version` / `tag` | 资产版本与 benchmark 标签（如 `benchmark-v0.1.0`） |

## 单资产目录结构（示例）

每条资产目录通常包含：

- `mesh/` — OBJ/GLB、碰撞 mesh、材质贴图
- `mjcf/` — MuJoCo 格式
- `urdf` — 可直接用于 SAPIEN / Isaac Gym / PyBullet
- `affordance/` — `affordance_annot.json`、`mesh_part_seg.glb` 等
- `video.mp4` — 资产预览

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [EmbodiedGen V2 实体页](../../wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md) | 论文报告的 **4K+ 跨格式资产库** 之公开下载入口 |
| [Manipulation](../../wiki/tasks/manipulation.md) | affordance + 仿真验证抓取位姿，服务操作 sim 资产上游 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **生成式 3D 资产** 与可仿真环境的数据飞轮 |

## 对 wiki 的映射

- 主挂 **`wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md`** 的「工程实践 / 数据集」节
- 仓库归档：**`sources/repos/embodiedgen.md`**

## 外部参考

- [HorizonRobotics/EmbodiedGenData](https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData)
- [Gallery Explorer Space](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer)
