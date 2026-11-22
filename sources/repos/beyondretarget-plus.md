# bear-ty/BeyondRetarget_plus（官方代码）

- **名称：** BeyondRetarget Plus
- **类型：** repo / offline-motion-generation / moving-camera / humanoid
- **URL：** <https://github.com/bear-ty/BeyondRetarget_plus>
- **项目主页：** [BeyondRetarget Plus](../sites/beyondretarget-plus.md)
- **基础版论文：** [BeyondRetarget（arXiv:2609.29850）](../papers/beyondretarget_arxiv_2609_29850.md)
- **基础版代码：** <https://github.com/bear-ty/BeyondRetarget>
- **许可证：** **CC BY-NC 4.0（非商业）**；TRAM、SPEC、SMPL、YOLO、数据集与模型资源各自遵循上游许可。
- **入库日期：** 2026-11-22
- **核对范围：** 完整 README（安装、Resources、Datasets、Inference、Training、Evaluation、License）、`scripts/infer_beyondretarget.py`、`scripts/prepare_robot_video_camera.py`、`scripts/extract_video_boxes.py`、`scripts/extract_video_coco17.py`、`scripts/postprocess_beyondretarget.py`、`lib/models/tram_robot.py`、`scripts/cache_tram_robot_features.py` 的入口与数据流。
- **通读：** 已完成
- **开源状态（2026-11-22）：** 训练、推理、相机前端处理与测试脚本可读；完整 Plus checkpoint 和机器人资产由官方共享 Drive 下载；SPEC/camcalib、部分 TRAM/PARE 文件及许可数据集需按上游说明补齐，不能视为仓库包含全部依赖。

## 一句话说明

BeyondRetarget Plus 基于冻结 TRAM/VIMO 人体视频骨干，学习机器人 DoF 适配与 static/contact 监督，利用可选的移动相机前端恢复世界坐标轨迹，并用完整 **C-RC1** 后处理产出可检查的机器人运动 `.npz`；用于 **离线** 数据生成，不直接等价于 base 版实时遥操作部署。

## 可复现入口（README + 源码核对）

| 步骤 | 文件 / 命令 | 产物或作用 |
|---|---|---|
| 环境 | `bash scripts/setup_environment.sh`、`python scripts/check_installation.py` | Conda 环境与依赖校验 |
| 人体跟踪 | `scripts/extract_video_boxes.py` | 原视频对齐的单人目标框 `track.npz` |
| 关节检测 | `scripts/extract_video_coco17.py` | 与同一 track 对齐的 `[T,17,3]` 图像关键点 |
| 相机前端 | `scripts/prepare_robot_video_camera.py --camera-mode moving` | SAM mask → TRAM 标定 / metric SLAM → `camera.npy` + 带内参 `boxes.npz` |
| 机器人推理 | `scripts/infer_beyondretarget.py` | `TramRobotAdapter.infer_sequence` → robot DoF、root、contact/static |
| 完整后处理 | `scripts/postprocess_beyondretarget.py` / `process_prediction` | C-RC1 重投影/足端修正，`final_motion.npz`、diagnostics |
| 特征缓存与训练 | `scripts/cache_tram_robot_features.py` → `scripts/train_beyondretarget.py` | 冻结 TRAM/VIMO 的 16 帧特征，再训练 robot heads |
| 静态头训练 | `scripts/generate_static_labels.py` → `scripts/build_true_world_static.py` → `scripts/train_static_head.py` | 生成标签并训练 static head（有额外外部 BEDLAM 标注约束） |
| 评测 | `scripts/emdb/run_cam.py`、`run_smpl.py`、`run_eval.py` 等 | EMDB 对照与 CPU 单元测试 |

**完整推理示意：**

```bash
python scripts/extract_video_boxes.py --video video.mp4 --weights assets/yolo/yolov8x.pt --output outputs/track.npz
python scripts/extract_video_coco17.py --video video.mp4 --boxes outputs/track.npz --weights assets/yolo/yolov8x-pose.pt --output outputs/coco17.npz
python scripts/prepare_robot_video_camera.py --video video.mp4 --boxes outputs/track.npz --output-dir outputs/frontend --camera-mode moving
python scripts/infer_beyondretarget.py --cfg configs/config_beyondretarget_motion_g1.yaml --video video.mp4 --boxes outputs/frontend/boxes.npz --checkpoint assets/checkpoints/beyondretarget_plus_8robots.pth --camera-trajectory outputs/frontend/camera.npy --keypoints outputs/coco17.npz --confirm-single-shot --output outputs/final_motion.npz
```

## 数据与边界

- **适配机型：** Unitree G1 / R1 / H1、Fourier GR1-T1 / GR2-V3、Booster T1、Tienkung、Atlas（官方列出的八种）。
- **C-RC1 输入约束：** 连续、单人、单镜头且至少 **16 帧**；frame、bbox、COCO17 和内参必须对齐原始视频坐标，轨迹中的缺失帧不能以插值 bbox 冒充真实观测。
- **外部依赖：** 视频相机前端需要 DROID-SLAM、SAM、ZoeDepth、SPEC/camcalib 及其权重；SMPL 模型、BEDLAM/H36M/3DPW/EMDB 数据需要各自获取。
- **输出判读：** `--raw-only` 仅中间结果；正常 `--output` 在 C-RC1 完成后才写入最终轨迹，并另保存 raw sidecar、诊断和阶段产物。
- **量化证据：** 当前 Plus 项目页尚未给出与 base 一致协议的完整定量表；**不能**把原论文 base 的 95.86% 成功率 / 192.8 ms 延迟写成 Plus 的结果。

## 关联 wiki

- [BeyondRetarget 基础论文与 Plus 扩展](../../wiki/entities/paper-beyondretarget-monocular-humanoid.md)
