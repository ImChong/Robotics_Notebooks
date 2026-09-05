# Tennis-Vision（HarshTomar1234/Tennis-Vision）

> 来源归档

- **标题：** Tennis-Vision — Tennis Detection and Visualization System
- **类型：** repo / computer-vision / sports-analytics / object-detection / tracking / homography
- **作者：** Harsh Tomar（个人；无独立机构项目页）
- **链接：** <https://github.com/HarshTomar1234/Tennis-Vision>
- **克隆：** `https://github.com/HarshTomar1234/Tennis-Vision.git`
- **Stars / Forks：** 67★ / 5（2026-09-05 GitHub API）
- **许可：** **MIT**（根目录 `LICENSE`；GitHub SPDX：MIT）
- **安装：** `pip install -e .` → `tennis-vision` CLI（Python 3.12 实测）
- **项目页：** 无独立 `*.github.io`；以 GitHub README + [`docs/public/TECHNICAL_OVERVIEW.md`](https://github.com/HarshTomar1234/Tennis-Vision/blob/main/docs/public/TECHNICAL_OVERVIEW.md) 为入口
- **入库日期：** 2026-09-05
- **一句话说明：** 单路广播相机网球分析管线：TrackNet 检球、ResNet-50 球场 14 点 + 单应、YOLOv8x+ByteTrack 选球员，再做击球/弹跳检测、发球速度与闭合弹道三维重建；**测不到的数一律拒报**。
- **为什么值得保留：** 为人形足球 / RoboCup / 体育 CV 提供 **第三人称广播视角** 对照，且把「单应只在拟合平面上有效」「重投影误差不能当有效性」「检测率 ≠ 定位精度」写成可复现的负结果。与本库 [Roboflow Sports](../../wiki/entities/roboflow-sports.md)、[场线检测](../../wiki/methods/soccer-field-line-detection.md)、[感知坐标后处理](../../wiki/concepts/perception-coordinate-postprocessing.md) 直接互补。
- **开源状态（2026-09-05 核查）：** **已开源** — 训练/推理/评测脚本齐全，409 个 pytest；`tennis-vision download-models` 拉约 140 MB 权重。TrackNet 球权重版权未声明、不随仓再分发（从原作者仓拉取）。可选 [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) 权重走 Meta SAM License，默认关闭。球场微调权重在 Hugging Face [`Coddieharsh/tennis-court-keypoints`](https://huggingface.co/Coddieharsh/tennis-court-keypoints)。完整 TrackNet / TennisCourtDetector 评测集 >7 GB、不可再分发，见仓内 `datasets/README.md`。
- **沉淀到 wiki：** 是 → [`wiki/entities/tennis-vision.md`](../../wiki/entities/tennis-vision.md)

## 仓库概况（2026-09-05 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`HarshTomar1234/Tennis-Vision`） |
| 默认分支 | `main` |
| 创建 / 最近推送 | 2025-03-14 / 2026-08-27 |
| Topics | `object-detection`, `player-stats`, `pytorch`, `shot-classification`, `yolov8` |
| 语言 | Python |
| Homepage | 空 |
| 硬件快照 | GTX 1050 Ti 4 GB 上 19 s 视频约 5 m 51 s（约 18.5× 实时） |

## README 入口（归纳）

| 组件 | 路径 / 命令 |
|------|-------------|
| 安装 | `pip install -e .` |
| 权重 | `tennis-vision download-models`（约 140 MB；TrackNet 需一次手动确认） |
| 分析 | `tennis-vision analyze <video> -o output/demo.avi`（`--max-frames 60` 冒烟） |
| 配置 | `configs/config.yaml`；开发缓存 `configs/dev.yaml`（默认关，避免串片） |
| 技术总览 | `docs/public/TECHNICAL_OVERVIEW.md`（每级：做什么 / 为何 / 算法 / 假设 / 失败） |
| 评测 | `eval/*.py`（README 每个数字都指到脚本） |
| 测试 | `pytest tests/`（409；CI 跑 `-m "not slow"`） |
| 球场关键点 | `court_line_detector/`（ResNet-50，14 点，逐帧） |
| 跟踪 | `trackers/tracknet_ball_tracker.py`、`trackers/player_tracker.py` |
| 几何 / 事件 | `utils/court_validity.py`、`hit_bounce_classifier.py`、`rally_decode.py`、`trajectory_3d.py` |
| 俯视 | `mini_visual_court/` |
| 概念笔记 | `notes/01_homography_basics.md` … `06_shot_detection.md` |

### 管线门（TECHNICAL_OVERVIEW 顺序）

1. 读视频 + **帧率支持门**（23–31 fps 才有证据）
2. YOLOv8x + ByteTrack 检人 → 全片六准则选两人 → **对侧网门**
3. TrackNet 热图取最大连通域
4. ResNet-50 14 点 → **场地线亮度支持门**（`MIN_LINE_SUPPORT=0.22`）
5. 地板单应；球只在弹跳/击球锚点上投影
6. y 反向 ∪ x 速度 ∪ 弹跳生成器 → 轨迹分类 → Viterbi 回合语法
7. 发球（过头 **且** 底线后）与物理证据过滤 Volley/Smash
8. 两已知接触之间的无自由参数抛物线三维重建

### 关键测得数字（README 自报，脚本可复现）

| 指标 | 结果 | 备注 |
|------|------|------|
| TrackNet 出点率 | 88.6% | **不是** 精度 |
| 可见球帧落入 5 px | 42.5% | 中位误差 5.4 px，p90 18.0 px |
| 端到端接触召回（25 clip） | 72.0% / 精 95.9% | 完美球坐标上界召回 87.6% |
| 击球/弹跳 held-out | 86.4% | 按 clip 划分；端到端 F1 选特征 |
| 球场关键点中位误差 | 2.90 px（微调后） | 2 211 张 TennisCourtDetector val |
| 发球速度 vs 转播雷达 | 比 0.96 | 两 clip；雷达读拍面、管线读落地 |
| 任意 15 s 转播窗 held-out | 14/14 **拒报** | 正确：窗口含切镜/回放 |

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 主实体 | [`wiki/entities/tennis-vision.md`](../../wiki/entities/tennis-vision.md) |
| 足球广播对照 | [`wiki/entities/roboflow-sports.md`](../../wiki/entities/roboflow-sports.md) |
| 场线 / 关键点方法 | [`wiki/methods/soccer-field-line-detection.md`](../../wiki/methods/soccer-field-line-detection.md) |
| 像素→场地坐标 | [`wiki/concepts/perception-coordinate-postprocessing.md`](../../wiki/concepts/perception-coordinate-postprocessing.md) |
| 机载场线定位对照 | [`wiki/queries/soccer-visual-field-localization-pipeline.md`](../../wiki/queries/soccer-visual-field-localization-pipeline.md) |
| 检测工程入口 | [`wiki/entities/ultralytics.md`](../../wiki/entities/ultralytics.md) |
| Kalman / RTS 负结果 | [`wiki/formalizations/kalman-filter.md`](../../wiki/formalizations/kalman-filter.md) |
| 姿态后端 | [`wiki/entities/mediapipe.md`](../../wiki/entities/mediapipe.md)、[`wiki/entities/sam-3d-body.md`](../../wiki/entities/sam-3d-body.md) |

## 参考链接

- 仓库：<https://github.com/HarshTomar1234/Tennis-Vision>
- 技术总览：<https://github.com/HarshTomar1234/Tennis-Vision/blob/main/docs/public/TECHNICAL_OVERVIEW.md>
- TrackNet 权重上游：[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)
- 球场微调权重：<https://huggingface.co/Coddieharsh/tennis-court-keypoints>
- 可选姿态：[facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- Ultralytics YOLOv8：<https://github.com/ultralytics/ultralytics>
