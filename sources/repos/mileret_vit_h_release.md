# ViT-H 官方实现仓库（mileret/ViT-H-release）

> 来源归档（ingest）

- **类型：** repo / humanoid control / video-to-humanoid / source code
- **主仓库：** <https://github.com/mileret/ViT-H-release>
- **项目页：** <https://awfuact.github.io/vit-h/> — [项目页归档](../sites/vit-h.md)
- **论文：** <https://arxiv.org/abs/2610.32966> — [论文归档](../papers/vit_h_arxiv_2610_32966.md)
- **许可证：** MIT（仓库 LICENSE，第三方资产各自遵循原许可）
- **入库日期：** 2026-10-27
- **复核日期：** 2026-11-08
- **核对范围：** 官方 README（安装、下载、数据、训练、推理、Release Plan）、STRUCTURE.md、PROVENANCE.md、docs/INFERENCE.md、g1_tracker/README.md；检查 scripts/train.sh、tools/train.py、tools/infer_video.py、tools/video_preprocess.py、g1_tracker/tools/track_g1.py 源码中的调用入口。
- **通读：** 已完成
- **一句话说明：** 已发布 G1 离线 video-to-humanoid 预训练、推理、MuJoCo 跟踪和部分权重；**在线遥操作与 DPO 后训练的代码尚未发布**。
- **沉淀到 wiki：** [ViT-H 论文实体页](../../wiki/entities/paper-vit-h-video-to-humanoid.md)

## 实际发布状态（以仓库 README 为准）

| 功能 | 状态 | 官方入口 |
| --- | --- | --- |
| 离线动作生成预训练 | 已发布 | `scripts/train.sh`、`tools/train.py`、Hydra 配置 |
| 离线 RGB 视频 → G1 参考动作 | 已发布 | `tools/infer_video.py`，预处理与离线 checkpoint |
| G1 运动跟踪仿真 | 已发布 | `g1_tracker/tools/track_g1.py`，MuJoCo + ONNX 追踪策略 |
| 离线生成 checkpoint、G1 ONNX policy | 提供下载 | [ViT-H-checkpoints](https://huggingface.co/datasets/eerrr1/ViT-H-checkpoints) |
| 离线训练预处理数据 | 提供下载，但需遵守数据授权 | [ViT-H dataset](https://huggingface.co/datasets/eerrr1/ViT-H) |
| 实机在线遥操作部署、在线生成预训练、DPO 后训练 | **尚未发布**，README 计划 2026-12-01 | Release Plan |

## 已核对的复现入口

### 离线训练

```bash
export VITH_DATASET_ROOT=/path/to/training_set
export NUM_DEVICES=8
export WANDB_MODE=offline
bash scripts/train.sh
```

`scripts/train.sh` 设置数据集/Hydra 训练参数并启动 `tools/train.py`；后者通过 Hydra 实例化 Lightning DataModule 与模型，调用 `Trainer.fit`。官方 release 实验采用 16 层 Transformer、1024 维隐空间、16 attention heads、120 帧窗口、混合来源 batch；预训练与论文 DPO 后训练是**不同**管线，不能把 `train.sh` 当作 DPO 入口。

### 离线视频推理

```bash
python tools/infer_video.py --video /path/to/input.mp4 \
  --ckpt /path/to/vith_offline_g1.ckpt \
  --sam3d-ckpt-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --sam3d-mhr-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --sam3d-root third-party/Fast-SAM-3D-Body \
  --output outputs/inference/input_g1.npz
```

`tools/infer_video.py` 在 `preprocessing_session` 内调用 `run_preprocess`、`load_data_dict`；`video_preprocess.py` 负责时间戳重采样到 30 fps、人体框、ViTPose 关键点、SAM3D 图像特征与相机估计（默认 SimpleVO，可选 DPVO 或已知静止相机）。随后模型滑窗生成 G1 29-DoF 参考动作，保存 `joint_pos`、`root_pos_w`、`root_quat_w` 等字段至 NPZ。

`docs/INFERENCE.md` 说明输出 `joint_pos` 为 BYD 关节顺序，而 `g1_tracker/README.md` 对 tracker 输入指定 IsaacLab 顺序。**跨入口使用前须验证并处理 DoF 轴序映射**，不能未经核实就将任意 NPZ 直接送入 tracker。

### MuJoCo 跟踪（独立入口）

```bash
python g1_tracker/tools/track_g1.py \
  --input outputs/example/motion.npz \
  --output outputs/example/tracked_motion.npz \
  --input-fps 30 \
  --policy-onnx inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx \
  --xml-path g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml \
  --grounding-xml g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml
```

`track_g1.py` 的 `main()` 将命令行参数传入 `vith_tracker.track_file`，读取参考轨迹并以 50 Hz 在 MuJoCo 中运行 ONNX 跟踪器，输出跟踪轨迹和 `is_success` 等字段。此路径验证的是**仿真跟踪**，不是实机部署。

## 复现边界与溯源

- 开发环境：Linux、Python 3.10、CUDA 12.1、PyTorch 2.3；还需取得 SAM3D/MHR、SMPL/SMPL-X 及可选 DPVO 等第三方模型/依赖。
- 训练数据只发布处理后的部分，原始 AMASS、BEDLAM、Human3.6M、BONES-SEED 等依各自授权取得；BONES-SEED 需要额外授权。
- 官方 `PROVENANCE.md` 说明 2026-08-06 训练时源码树并非 Git 仓库，无法证明 release 与训练启动时源码逐字节完全一致；不能夸大为 bit-for-bit 复现。
- 本条核对的是代码、README 和入口逻辑；**没有实际运行训练、推理、仿真或真机**。

## Wiki 映射

- [ViT-H 论文实体页](../../wiki/entities/paper-vit-h-video-to-humanoid.md)
