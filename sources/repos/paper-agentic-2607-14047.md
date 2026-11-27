# open-gigaai/Zero2Skill（官方仓库核对归档）

- **仓库地址：** https://github.com/open-gigaai/Zero2Skill
- **协议：** Apache-2.0（含 `LICENSE` 与 `NOTICE`）
- **默认分支：** `main`
- **开源状态：** 部分开源 — 仓库以 Apache-2.0 公开了第一方采集/抓取管线、运动控制辅助脚本、数据质检/统计平台及 4 个 Agent Skills；明确不重分发专有 AnyGrasp 运行时（`.so`、license、checkpoint）、SAM3 官方环境及外部 HDF5 录制脚本 `COLLECT_RECORDER_PY`。
- **核对范围：** 核对仓库元信息（`full_name`、`description`、`license`、`default_branch`）、文件样本清单（`files_sample`）与 `README.md` 预览（`readme_preview`）。
- **复现边界：** 静态代码核对，未本地运行。

## 目录结构与关键入口核对

- **对 wiki 的映射：** [Zero2Skill 实体页](../../wiki/entities/paper-agentic-2607-14047.md)
- **环境与硬件配置（`configs/` & `docs/`）：**
  - `configs/paths.env.example`、`configs/load_paths.sh`、`configs/camera_extrinsics.example.json`：配置 `CONDA_ENV=zero2skill`、`CONDA_ENV_SAM3`、`CONDA_ENV_ANYGRASP`、机械臂驱动、相机启动、ROS topic、内参 `CAM_FX..` 与手眼外参。
  - `docs/architecture.md`、`docs/setup.md`、`grasp-tools/ANYGRASP_RUNTIME.md`：架构说明与独立环境配置文档。
- **Agent Skills 编排层（`skills/`）：**
  - `skills/self-learning/SKILL.md`、`skills/self-learning/analyze_result.example.yaml`、`skills/self-learning/reference/OOD_handling_flow.md`：薄编排器与任务规划文件 `analyze_result.yaml` 模板及 OOD 暂停上报流程。
  - `skills/1-analyze-task/SKILL.md`：两阶段流程入口（Phase A 从指令与场景生成采集/重置计划；Phase B 运行重试上限 $N=3$ 的验证门控循环并在 Alert 时通过自然语言更新配置/判据）。
  - `skills/grasp-tool/SKILL.md`、`skills/grasp-tool/references/grasp_experience.md`、`skills/grasp-tool/scripts/start_robot_infra.sh`：抓取工具技能、可阅读操作经验规则与基础设施启动脚本。
  - `skills/understand-three-view-images/SKILL.md` 及 `scripts/`（`_env.sh`、`auto_capture_images.sh`、`auto_capture_rgb_depth.sh`、`auto_understand_images.sh`、`auto_understand_images_bbox.sh`、`capture_and_understand_one_view.py`、`capture_one_view_image.py`、`capture_three_view_images.py`、`capture_three_view_rgb_depth.py`、`understand.py`、`visualize_bbox.py`）：三视角捕获与 VLM（`ARK_API_KEY`）场景理解/成败判定脚本。
- **抓取与采集运行管线（`grasp-tools/`）：**
  - `grasp-tools/run_multi_pipeline_recorder.sh`：主采集/抓取入口（示例 `--mode diligent --no-gui --top-down --task "orange" right 0.3 0.27 0`）。
  - `grasp-tools/run_seg.py`、`start_seg_server.sh`、`stop_seg_server.sh`：SAM3 文本分割服务端与调用脚本。
  - `grasp-tools/run_grasp.py`、`run_grasp_mask.py`、`joint_grasp_plan.py`、`piper_pose_ik.py`、`joint_grasp_publisher_npy.py`、`joint_publisher.py`、`eef_publisher_npy.py`：AnyGrasp 6-DoF 规划、掩码过滤、Pinocchio IK 求解与关节发布工具。
  - `grasp-tools/capture_front_rgb_depth.py`、`auto_capture_rgb_depth.sh`：头戴相机对齐 RGB-D 采集。
- **数据质检、离线重判与训练准备（`grasp-tools/collect/`）：**
  - `episode_stats.py`、`data_quality_check.py`、`stats_report.py`、`serve_report.py`、`report_template.html`、`render_episode_video.py`：每回合运动平滑度/图像质量统计、异常评级（good/marginal/bad）与远程 HTML 报告生成。
  - `judge_episode_offline.py`、`prepare_training_set.py`、`rescue_data.sh`：基于末帧图像的离线 VLM 统一重判及固化 manifest 训练集打包。
