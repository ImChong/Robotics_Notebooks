# roboflow/sports

> 来源归档

- **标题：** sports（computer vision and sports）
- **类型：** repo / computer-vision / sports-analytics / object-detection / keypoint-detection
- **机构：** 罗博福流（Roboflow）
- **链接：** <https://github.com/roboflow/sports>
- **Stars / Forks：** ~5.2k★ / —（2026-07-27 核查）
- **许可：** **MIT**（`sports` 分析与可视化代码；依赖 [Supervision](https://github.com/roboflow/supervision) 同为 MIT）
- **安装：** 尚无独立 PyPI 正式版；`pip install git+https://github.com/roboflow/sports.git`（Python ≥3.8）
- **项目页：** 无独立 `*.github.io`；以 GitHub README + `examples/soccer/` 为入口
- **入库日期：** 2026-07-27
- **一句话说明：** Roboflow 开源的 **体育 CV 工具与足球分析 demo**：可复用的球场配置 / 单应变换 / 球跟踪 / 球队聚类，叠加 YOLOv8 检测与 Supervision 标注，打通「检测 → 跟踪 → 俯视雷达」管线。
- **为什么值得保留：** 为人形足球 / RoboCup 感知提供 **第三人称广播视角** 对照实现——尤其是 **球场关键点 → 单应 → 俯视坐标** 与小目标球跟踪；与本库 [场线检测](../../wiki/methods/soccer-field-line-detection.md)、[Ultralytics](../../wiki/entities/ultralytics.md) 直接互补。
- **开源状态（2026-07-27 核查）：** **已开源** — 库代码 + soccer 示例完整可跑；预训练 `.pt` 与样例视频经 `examples/soccer/setup.sh`（`gdown`）从 Google Drive 拉取；Universe 上另有球员/球/球场关键点等数据集。
- **沉淀到 wiki：** 是 → [`wiki/entities/roboflow-sports.md`](../../wiki/entities/roboflow-sports.md)

---

## 仓库结构（入库快照）

| 路径 | 作用 |
|------|------|
| `sports/configs/soccer.py` | `SoccerPitchConfiguration`：标准球场尺寸（cm）与顶点/边拓扑 |
| `sports/common/view.py` | `ViewTransformer`：`cv2.findHomography` 图像↔球场平面 |
| `sports/common/ball.py` | `BallTracker`（缓冲质心选检）+ `BallAnnotator` |
| `sports/common/team.py` | `TeamClassifier`：SigLIP 嵌入 → UMAP → KMeans（两队） |
| `sports/annotators/soccer.py` | `draw_pitch` / 俯视点标注等 |
| `examples/soccer/main.py` | 六模式 CLI：PITCH / PLAYER / BALL / TRACKING / TEAM / **RADAR** |
| `examples/soccer/notebooks/` | 球员/球/球场关键点 YOLOv8 训练 Colab |

**主依赖（`setup.py`）：** `supervision`、`numpy`、`opencv-python`、`transformers`、`umap-learn`、`scikit-learn`；soccer 示例另需 **Ultralytics YOLOv8**（**AGPL-3.0**，与本仓 MIT 分离——demo README 已写明）。

---

## 挑战清单（README 归纳）

1. **球跟踪** — 小目标、高速运动、高分辨率下易丢检  
2. **球衣号码 OCR** — 模糊、遮挡、背对镜头  
3. **球员跟踪** — 频繁遮挡导致 ID 切换  
4. **球员再识别** — 出画再入画、移动机位、外观相似  
5. **相机标定** — 动态机位下仍要支撑速度/跑动距离等高级统计  

篮球侧另挂球场关键点与球衣号码 OCR 的 Universe 数据集（库代码当前以足球 demo 为主）。

---

## Soccer demo 模式

| Mode | 作用 |
|------|------|
| `PITCH_DETECTION` | 球场边界 / 关键点 |
| `PLAYER_DETECTION` | 球员、门将、裁判、球 |
| `BALL_DETECTION` | 球检测 + `BallTracker` |
| `PLAYER_TRACKING` | 跨帧 ID（Supervision tracker） |
| `TEAM_CLASSIFICATION` | SigLIP + UMAP + KMeans 分队上色 |
| `RADAR` | 关键点单应 + 检测/跟踪/分队 → **俯视雷达图** |

最短路径：`pip install git+https://github.com/roboflow/sports.git` → `cd examples/soccer && pip install -r requirements.txt && ./setup.sh` → `python main.py --source_video_path data/2e57b9_0.mp4 --target_video_path out.mp4 --device cpu --mode RADAR`。

---

## 对 wiki 的映射

- 主实体：[`wiki/entities/roboflow-sports.md`](../../wiki/entities/roboflow-sports.md)
- 场线 / 关键点方法：[`wiki/methods/soccer-field-line-detection.md`](../../wiki/methods/soccer-field-line-detection.md)
- 视觉定位流水线对照：[`wiki/queries/soccer-visual-field-localization-pipeline.md`](../../wiki/queries/soccer-visual-field-localization-pipeline.md)
- 检测工程入口：[`wiki/entities/ultralytics.md`](../../wiki/entities/ultralytics.md)
- 任务语境：[`wiki/tasks/humanoid-soccer.md`](../../wiki/tasks/humanoid-soccer.md)
- 同机构检测模型：[`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)
