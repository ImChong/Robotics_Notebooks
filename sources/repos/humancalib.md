# HumanCalib：以行走人体完成多相机外参标定

> 官方仓库一手资料归档；本页为知识摘录、验证说明与源码索引，不全文转载。

- **标题：** HumanCalib — Extrinsic Camera Calibration from a Walking Person
- **类型：** repo / 多相机外参标定
- **代码：** https://github.com/flodelaplace/HumanCalib
- **主文档：** https://github.com/flodelaplace/HumanCalib/blob/main/README.md
- **技术方法：** https://github.com/flodelaplace/HumanCalib/blob/main/docs/METHOD.md
- **命令与配置：** https://github.com/flodelaplace/HumanCalib/blob/main/HOWTO.md
- **论文预印本（作者链接）：** https://hal.science/hal-05855397v1 （官方称在审，未作为已发表论文）
- **源码核对版本：** `flodelaplace/HumanCalib@bb5dee70c1e783f2af74f70e0cb2d3e90ba9f99d`（2026-11-22 检索）
- **核对范围：** 官方 README、docs/METHOD.md、HOWTO.md 的输入、两种姿态引擎、人员选择、线性初始化、外点剔除、BA、尺度/竖直估计、评估、运行与许可章节；阅读 `src/humancalib/cli.py` 的 `run_pipeline`、`pipeline/run_calib_linear.py`、`pipeline/reselect_person.py`、`pipeline/run_ba.py`、`pose/metrabs_inference.py`、`postprocessing/scale_scene.py`、`postprocessing/evaluate_calibration.py` 的入口与主要实现；未逐文件审计全部仓库，未在本环境实际运行模型与数据集。
- **通读：** 已完成
- **许可证：** HumanCalib 代码 MIT；MeTRAbs 预训练权重含非商业使用限制，VideoPose3D 代码与权重 CC BY-NC 4.0；RTMPose 权重的上游授权需另行核对。
- **映射：** [HumanCalib 独立实体](../../wiki/entities/humancalib.md)

## 官方公开的能力与边界

- 输入：多台**固定相机**的同步视频（每路一个）、每台相机的**已知内参及畸变系数**（Pose2Sim `Calib_scene.toml`），以及参与者身高（米）。视频文件名须与 TOML 相机分节对应。
- 不用棋盘格或标定杆求**外参**，但不取代相机内参标定，也不负责视频同步；走动者应在多视角间有足够共同可见区域和空间覆盖。
- 默认 MeTRAbs 一次预测相机坐标下 2D/3D 人体姿态；可选 RTMPose + VideoPose3D 先 2D 检测再时间序列 3D 提升。
- 人员关联：基于腿部摆动筛选正在行走的人，之后使用重投影几何重选、避免旁观者影响。
- 线性初始化：不同相机系人体骨架经 Procrustes / Umeyama 对齐，选择参考相机；对错帧剔除后执行带重投影和骨骼正则的 bundle adjustment。
- 竖直：跨整个步行序列统计头到足方向，并消除步行方向引入的前倾；尺度：MeTRAbs 通过头部虚拟标记至脚底的步行距离统计结合真实身高；RTMPose 使用身段比例。原点及水平方向由参考帧的足部关键点确定。
- 结果：`output/<session>/results/Calib_scene_calibrated.toml`（带姿态、米制比例和重力对齐坐标系的 Pose2Sim 格式）；可生成可视化、重投影误差诊断。

## 官方验证（README 报告，非本次复测）

- 官方报告在 BioCV、I-MOVE-23、OpenCap、LBMC、COMFI 五套公开数据的 97 个 trial 测试默认 MeTRAbs；报告 0 次标定失败。
- 有参考标定的三套数据：相对相机旋转误差中位数分别为 BioCV **0.47°**、I-MOVE-23 **0.51°**、OpenCap **1.89°**；MRE 中位数 **2.75 / 3.34 / 1.69 px**，不同相机内参及分辨率下不能直接比较 px。
- 用于尺度规则设计之外的四套数据上，报告尺度中位绝对误差约 **1.1%**。RTMPose 路径报告 97 次中有 1 次失败、尺度误差约 1.7%，并非与 MeTRAbs 完全等价。
- 厂商式“无标定”理解不准确：内参焦距误差 5% 可基本传递为尺度误差，给定身高每差 1 cm 官方估计尺度约差 0.6%；重投影残差低也不直接保证外参准确。
- 这些是项目 README 及方法说明中的作者报告；本次未在本地复现。预印本尚未经过同行评审。

## 源码入口核对

| 执行环节 | 实现 / 入口 |
|---|---|
| CLI 主流程 | `src/humancalib/cli.py::run_pipeline`（`humancalib run`） |
| 姿态提取 | `pose/metrabs_inference.py` 或 `pose/rtmlib_inference.py`；后者由 `pose/lifting.py` 补 3D |
| 相机内参及会话 | `pipeline/create_cameras_from_toml.py`、`pipeline/write_session.py` |
| 人员筛选与重选 | `pipeline/reselect_person.py`、`pipeline/motion_selection.py` |
| 初始化与异常帧 | `pipeline/run_calib_linear.py` → `calibration/calib_linear.py`、`pipeline/detect_outlier_frames.py` |
| BA | `pipeline/run_ba.py` 在子进程调用 `calibration/ba.py`，OOM 时提高取帧间隔重试 |
| 后处理 | `postprocessing/evaluate_calibration.py` → `postprocessing/scale_scene.py`，导出 TOML |

运行命令示例来自 HOWTO（身高示例须换为实际值）：

```bash
humancalib run input/my_session input/my_session/Calib_scene.toml output/my_session --height 1.84
```

官方文档同时给出 Docker 和 Python API `humancalib.calibrate(..., height=...)`。GPU 姿态推理需 NVIDIA GPU，权重依赖额外下载。

## 对 wiki 的映射

- [HumanCalib：走路标定多相机外参](../../wiki/entities/humancalib.md)
- [FreeMoCap：低成本多相机动捕](../../wiki/entities/freemocap.md)
- [动作重定向：Stage 2 数据源与质量](../../roadmap/depth-motion-retargeting.md)
