# EgoMo3R 官方源码与运行入口

> 官方仓库与源码静态审阅归档（不等于独立复现）

- **类型：** repo / egocentric / 3d-scene-reconstruction / human-motion
- **主仓库：** https://github.com/chaitanya100100/EgoMo3R
- **独立运动扩散先验：** https://github.com/chaitanya100100/motion_diffusion
- **预训练权重：** https://huggingface.co/chaitanya100100/egomo3r-motion-prior
- **论文：** https://arxiv.org/abs/2611.01627（[论文归档](../papers/egomo3r_arxiv_2611_01627.md)）
- **项目：** https://chaitanya100100.github.io/EgoMo3R/（[官网归档](../sites/egomo3r_project.md)）
- **核对日期：** 2026-11-15
- **核对范围：** 主仓 README、run/run_all_stages.sh、run/optimize.py、model/motion_prior.py、config/defaults.py、config/stage1–4.yaml；motion_diffusion README 的安装、权重及训练入口。
- **通读：** 已完成
- **沉淀到 wiki：** [EgoMo3R 论文实体](../../wiki/entities/paper-egomo3r.md)

## 运行路径（基于实际文件）

1. [motion_diffusion README](https://github.com/chaitanya100100/motion_diffusion#installation)：创建 Python 3.10 的 modiff 环境、按 CUDA 安装 PyTorch、安装 SMPL-X；从 [Hugging Face](https://huggingface.co/chaitanya100100/egomo3r-motion-prior) 下载 motion-prior checkpoint / normalization statistics；sanity_check 的示例期望头部位置误差约 2.5 cm（为官方示例值，未在此运行）。
2. 在相同环境运行主仓 `pip install -r requirements.txt`；`bash checkpoints/download_sam2.sh` 下载 SAM2 权重；在 `config/defaults.py` 配置 `motion_prior_path`。
3. 入口 `bash run/run_all_stages.sh [sample_name] [dataset] [exp_name]` 按 1→4 执行 `python run/optimize.py config ./config/stageN.yaml`。缓存几何先验，遇已有 stage 输出默认跳过。
4. `run/optimize.py`：`prepare_batch` → `compute_flows` (RAFT) / `compute_depths` (MoGe-2) / `compute_tracks` (CoTracker3) / `compute_segmentation` (Grounding DINO+SAM2) → `get_motion_prior` → 按奇偶 stage 调用场景或动作重建 → `save_results`。
5. 奇数阶段的 `scene_reconstruction_optim` 用 `compute_flow_loss`、`compute_track_loss`，stage3 才打开深度尺度优化与地面 `compute_floor_loss`；偶数阶段的 `motion_reconstruction_optim` 调用 `MotionPrior.sample_motion_2`（stage2，未启用 guidance）或 `MotionPrior.sample_motion_with_guide`（stage4，开启 guidance）。
6. 最终 `python run/visualize.py config ./config/stage4.yaml inputs.sample_name iiith_cooking_58_2___0___8` 查看交互式 3D 输出。

## stage 对照

| stage | 代码配置 | 实际职责 |
|------|----------|----------|
| 1 | config/stage1.yaml | 场景：1000 iter，优化相机，不优化深度尺度，不加 floor loss |
| 2 | config/stage2.yaml | 动作：通过条件扩散 `sample_motion_2`，`use_guidance: false` |
| 3 | config/stage3.yaml | 场景：1000 iter，联合优化相机与尺度，启用 floor loss |
| 4 | config/stage4.yaml | 动作：`sample_motion_with_guide`，启用基于轨迹/滑动项的 guidance |

## 输出与适用限制

- `outputs/<dataset>/<sample>/` 保存 `depths/`、`flows/`、`tracks/`、`object_segmentation/` 缓存，实验子目录含每阶段相机轨迹、深度尺度与 SMPL 参数文件。
- **相机适配：** `model/inputs.py` 中 Project Aria RGB 内参固定为 512×512、f=150 px；其他摄像头必须修改 `load_aria_intrinsics` 和输入去畸变处理。
- **时间采样：** 运动先验按 10 FPS、80 帧/8s 窗口训练；长视频采用 sliding-window sampling，其他 FPS 不在当前预设支持范围。
- **可复现边界：** 开放了研究推理优化入口与示例；依赖 SAM2、SMPL-X 单独下载/许可及 motion prior checkpoint；本次仅静态阅读原始源码，**未实测完整 GPU 重建**。
