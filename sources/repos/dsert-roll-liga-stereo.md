# mickeykang16/LIGA-DSERT-RoLL — 双目 3D 检测官方基线

- **仓库：** https://github.com/mickeykang16/LIGA-DSERT-RoLL
- **论文/项目：** https://arxiv.org/abs/2604.03685 · https://jeongyh98.github.io/dsert-roll/
- **定位：** 将 LIGA-Stereo 复现代码移植到 DSERT-RoLL 的 RGB / Thermal / Event 双目检测评测；**不是**论文所提全模态融合网络的同一实现。
- **数据预处理：** Event 输入先以 `tools/gen_event_voxel.py` 生成 16-bin event voxel；通过 `python -m liga.datasets.dsert.lidar_dsert_dataset create_dsert_infos --modality rgb|thermal|event` 生成数据索引。这与主 3DOD 代码的 **5-bin** 配置不同，不可混用。
- **训练入口：** `./scripts/dist_train.sh {GPUS} only-lidar.pseudo_lidar configs/stereo/dsert_models/only-lidar.yaml`，随后使用第一阶段 LiDAR checkpoint 训练 `liga_rgb.yaml`、`liga_thermal.yaml` 或 `liga_event_new_anchor_bin16.yaml`。
- **评测入口：** `./scripts/dist_test_ckpt.sh {GPUS} configs/stereo/dsert_models/liga_rgb.yaml /path/to/checkpoint.pth data/dsert/dsert_infos_val_rgb.pkl`（thermal/event 替换对应 config 与 pkl）；`tools/test.py` 推理，`tools/eval_dsert.py` 计算 Waymo 风格指标。
- **依赖边界：** README 指训练为 Python 3.7 / PyTorch 1.6 / CUDA 10.1；Waymo-style 评测需单独 Python 3.6 TensorFlow 环境。未在本次环境实跑。
- **许可：** 补充材料 §1 指向 CC BY-NC 4.0 研究用途，第三方 LIGA/spconv 另按各自许可证。
- **关联：** [数据集](./dsert-roll-dataset.md) · [3DOD 多模态框架](./dsert-roll-3dod.md) · [主实体](../../wiki/entities/paper-dsert-roll.md)
