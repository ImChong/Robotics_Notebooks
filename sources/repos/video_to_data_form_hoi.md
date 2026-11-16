# nvidia-isaac/video_to_data（FORM-HOI / MV-HOI 官方管线）

> 来源归档（ingest）

- **标题：** nvidia-isaac/video_to_data（FORM-HOI / MV-HOI 多视角人–物重建与导出管线）
- **类型：** repo / 4d-reconstruction / multiview-hoi / video-to-data
- **链接：** <https://github.com/nvidia-isaac/video_to_data>
- **维护：** NVIDIA Isaac（NVIDIA CORPORATION & AFFILIATES）
- **许可：** 仓库中工作流与导出核心模块头部标注 `SPDX-License-Identifier: Apache-2.0`（其余子包与权重遵循对应仓内与外部模型许可，如 SAM3D Body 需接受 Hugging Face 门控协议）
- **入库日期：** 2026-11-14
- **一句话说明：** NVIDIA Isaac 开源的 Video-to-Data（V2D）框架，其中 `docs/form-hoi/` 与 `reconstruction/` 下的 MV-HOI 标定、多视角 4D 人–物重建及生产级 OSMO 质检导出管线直接生成 [FORM-HOI](../../wiki/entities/dataset-form-hoi.md) 数据集，`robotic_grounding/` 承接 `soma_params.npz` 等产物的机器人运动重定向。
- **开源状态：** **已开源（本地单序列重建可运行；生产批处理导出需内部 OSMO/CSS/HITL/PostgreSQL 基础设施）**
- **核对范围：** 2026-11-14 基于稀疏检出（sparse-checkout）的 `nvidia-isaac/video_to_data` 本地快照进行源码核对，实际阅读：`README.md`、`docs/form-hoi/README.md`、`reconstruction/CLAUDE.md`、`reconstruction/README.md`（包总览）、`reconstruction/docs/mv_hoi_local_pipeline.md`、`reconstruction/modules/v2d_pipelines/{run_mv_calibration.py,run_mv_hoi_reconstruction.py}`、`reconstruction/workflows/mv_hoi/{README.md,orchestration/export.py,osmo/mv_hoi_reconstruction.yaml,osmo/mv_hoi_export.yaml}`、`reconstruction/modules/v2d_mv_postprocess/lib/{finalize_export.py,export_sequence.py}` 及 `robotic_grounding/README.md`（Whole-body SOMA→G1 章节）。注：该稀疏检出快照未拉取 `robotic_grounding/scripts/retarget/soma_to_g1.py` 源文件本体。
- **通读：** 已完成
- **沉淀到 wiki：** [FORM-HOI](../../wiki/entities/dataset-form-hoi.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|---|---|
| `docs/form-hoi/README.md` | **已开源说明** — 明确指出 FORM-HOI 序列由 `video_to_data` 的 MV-HOI 重建管线产出，并指向 `reconstruction/docs/mv_hoi_local_pipeline.md` |
| 本地多视角管道（`v2d_pipelines`） | **已开源可运行** — 宿主机 Python 仅装零 ML 依赖的 `docker` 包装器，编排 `v2d_rosbag`、`v2d_mv_calibration` 等共 11 个容器镜像完成标定与单序列 18 步重建 |
| 生产 OSMO 与质检导出（`workflows/mv_hoi/`） | **源码公开、运行依赖内部基础设施** — OSMO YAML 与 `export.py`、`finalize_export.py` 逻辑全开源，但生产运行需连接 PostgreSQL、CSS S3/Swift、HITL AWS S3 及 Kratos DRS 凭证 |
| 下游机器人重定向（`robotic_grounding/`） | **入口文档已核对** — `robotic_grounding/README.md` 说明用 `scripts/retarget/soma_to_g1.py` 读取 `soma_params.npz`、`poses.npy` 与 `reconstructed_mesh/output_aligned.glb` 重定向至宇树 G1（本轮稀疏检出未含该 `.py` 本体） |

## 核心源码与架构核查

1. **宿主编排 + 容器推理（Host / Container Split）与文件契约：**
   - 依 `reconstruction/CLAUDE.md`，每个 `v2d_<module>` 拆分为仅装容器内重度 ML 依赖的 `lib/` 与仅含 `docker run` 封装的 `docker/`；跨模块通过磁盘文件与 `v2d_common` 数据类（`DepthImage`、`CameraIntrinsics`、`Transform3d`、`BoundingBox`、`Mask`、`mv_config` / `edex`）交换数据。
   - 硬件采集基于 **NVIDIA Nova Orin + 4 组 Leopard Imaging Hawk 立体相机**（左右共 8 路视频流，默认 1920×1200），录制为 `.mcap` ROS bag + `hoi_metadata.yaml` + 对象网格（如 `output_aligned.glb` 与可选对称文件 `output_symmetry.json`）。`export_sequence.py` 按 `LEFT_CAMERAS`（`front/back/left/right_stereo_camera_left`）导出 4 路深度与人/物掩码，以及 RGB / 匿名化 RGB（支持 `ffv1_sidecar` 等存储格式）。
2. **两阶段本地主入口（`v2d.pipelines`）：**
   - **标定入口 `run_mv_calibration.py`：** 先调用 `run_rosbag_to_edex(..., no_extrinsics=True)` 解包图像与内参，再调用 `run_calibrate_extrinsics(..., calibration_setup="stereo4_6x10_100mm_marker")`（另支持 `stereo4_6x10_22p58mm_marker`）输出标定后的 `extrinsics/edex` 与 `calibration_accuracy.json`。
   - **重建入口 `run_mv_hoi_reconstruction.py`：** 按顺序串行调用 18 个容器包装函数：
     1. `run_rosbag_to_edex`（解包序列图像与内参）→ `run_mv_preprocess`（校正、缩放、视频编码、`hoi_metadata.yaml` 生成 `prompt.txt` 与对齐网格）；
     2. `run_mv_detect_and_blur_faces`（生成 `face_detector/videos/` 脱敏预览流，不修改供下游视觉模块读取的 RGB 流）；
     3. `run_mv_image_list_to_depth`（FoundationStereo 估计双目深度）；
     4. 物体流：依据 `_read_object_bbox_source` 选择 `labeled_bboxes` 或调用 `run_mv_image_list_to_object_bboxes`（Grounding DINO 读取 `prompt.txt`）→ `run_mv_videos_to_masks`（SAM2 物体掩码）→ `run_mv_videos_to_poses`（FoundationPose 结合深度、掩码与 `_find_pinned_mesh`、`output_symmetry.json` 跟踪 6-DoF 位姿）；
     5. 人体流：`run_mv_track_bboxes`（Detectron2 人体框追踪）→ `run_mv_videos_to_masks`（SAM2 人体掩码）→ `run_mv_optimize_mhr_params`（SAM3D Body 多视角优化输出 `mhr_params_mv.pt` 与 `mhr_mesh_mv.pt`）→ `run_export_soma`（将 MHR 参数导出为 `soma_params.npz`）；
     6. 几何、评估与可视化后处理：`run_mv_export_fused_pointcloud`、`run_mv_estimate_ground_plane`、`run_mv_eval_chamfer_human`、`run_mv_eval_chamfer_object`、`run_mv_eval_silhouette_mask_human`、`run_mv_eval_silhouette_mask_object`、`run_mv_render_hoi_overlay`（读取脱敏视频渲染叠加视图）与 `run_mv_visualize_wis3d`。
3. **本地诊断脚本与 FORM-HOI 生产发布链的关键边界：**
   - `mv_hoi_local_pipeline.md` 明确提示：本地 `run_mv_hoi_reconstruction.py` **不执行**生产级 `check_accuracy`、`upload_hitl`、Kratos DRS 人工 QC 查表、接触窗口裁剪与最终提交导出，且本地运行器不支持断点跳过（失败需换用新的输出目录全量重跑）。
   - 生产级 OSMO 工作流 `mv_hoi_reconstruction.yaml` 在前述各任务外增设 `validate_labeled_bboxes`、`check_object_mask`（`--report-only`）、聚合门禁 `check_accuracy`（默认配置如 `max_chamfer_object: 40.0`、`max_chamfer_human: 40.0`、`min_silhouette_bbox_containment: 0.8`）与 `upload_hitl`；人工审核完成后由 `orchestration/export.py` 调度 `mv_hoi_export.yaml` 执行 `v2d.mv.postprocess.lib.finalize_export`：完成 `detect_interaction_trim` / `prepare_trimmed_source` 接触裁剪、`export_sequence` 转码、合并人工与指标异常区间写入 `failure_segments.json` 与 `interaction_trim.json`、执行 `evaluate_post_trim_qc` 与 `validate_export_payload` 校验并写入 `commit.json`（`v2d.mv_hoi.export_commit.v3`）。
4. **GPU 与环境约束：**
   - 本地 CUDA 核校验明确限定 compute capability 为 `{(8, 0), (8, 6), (8, 9), (9, 0)}`（`sm_80` / `sm_86` / `sm_89` / `sm_90`，不支持 Blackwell `sm_120`）；OSMO 生产管线在 NVIDIA H100、L40、L40S 上完成验证，TensorRT 引擎按目标架构匹配构建。RTX 4090 热启动跑完一段约 30 秒序列大约耗时 1 小时。

## 对 wiki 的映射

- [FORM-HOI](../../wiki/entities/dataset-form-hoi.md)
