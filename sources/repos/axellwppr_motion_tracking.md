# axellwppr_motion_tracking

> 来源归档（ingest）

- **标题：** Axellwppr/motion_tracking — Whole Body Motion Tracking
- **类型：** repo
- **来源：** GitHub
- **链接：** https://github.com/Axellwppr/motion_tracking
- **Stars：** ~268（2026-05-19 快照）
- **入库日期：** 2026-05-19
- **一句话说明：** 基于 [GentleHumanoid](https://gentle-humanoid.axell.top/) 与 **mjlab** 的 Unitree G1 全身运动跟踪训练/评估/部署仓：通用高动态 tracking、上半身柔顺交互、独立 VR 遥操作栈与 `sim2real` ONNX 导出。

## 核心摘录

### 定位与能力

- 在 GentleHumanoid 代码基上提供 **训练、评估、部署** 全套资产；仿真与训练后端为 **mjlab**（非 Isaac Lab）。
- 训练目标：**通用、鲁棒、高动态** 全身 motion tracking；同时保留 GentleHumanoid 的 **compliance-aware** 上半身交互能力。
- **VR 实时遥操作** 由独立 teleop 栈支持（README 指向 `sim2real` 与 VR motion source 配置）。
- 预训练策略演示站：[motion-tracking.axell.top](https://motion-tracking.axell.top)（单模型跨多样高动态动作泛化）。

### 数据管线

- 推荐直接下载预处理数据集（Google Drive），目录结构 `dataset/amass_all/`、`dataset/lafan_all/` 等。
- 自建数据：用 [Axellwppr/GMR](https://github.com/Axellwppr/GMR) 将 AMASS / LAFAN 重定向为 npz（`fps`, `root_pos`, `root_rot`, `dof_pos`, `local_body_pos`, `local_body_rot`, `body_names`, `joint_names`），再跑 `generate_dataset.sh`。
- 与论文一致：GentleHumanoid 训练语料亦经 **GMR** 重定向 AMASS、InterX、LAFAN 等（约 25 小时、50 Hz，过滤过高动态非交互片段）。

### 训练与部署

- 依赖管理：**uv**（`uv sync`）。
- 训练：`bash train.sh`（可配 WandB）；标准设置约 **4× A100、~15 小时**。
- 评估 / 导出：`uv run scripts/eval.py --run_path ${wandb_run_path} -p [--export]` → ONNX 写入 `scripts/exports/`，再拷至 `sim2real/assets/ckpts/` 并改 `sim2real/config/tracking.yaml`。
- 部署测试流程见 `sim2real/README.md`（sim2sim / sim2real、UDP motion selector、VR motion source）。

## 对 wiki 的映射

- [GentleHumanoid（上半身柔顺运动跟踪）](../../wiki/methods/gentlehumanoid-motion-tracking.md)
- [motion_tracking 工程实体](../../wiki/entities/axellwppr-motion-tracking.md)
- [GMR（运动重定向）](../../wiki/methods/motion-retargeting-gmr.md)
- [mjlab](../../wiki/entities/mjlab.md)
