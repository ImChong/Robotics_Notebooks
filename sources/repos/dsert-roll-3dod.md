# Chohoonhee/DSERT-RoLL-3DOD — 官方多模态 3D 检测实现

- **官方仓库：** https://github.com/Chohoonhee/DSERT-RoLL-3DOD
- **论文/项目：** https://arxiv.org/abs/2604.03685 · https://jeongyh98.github.io/dsert-roll/
- **开放状态：** 检测模型代码、配置、评测运行脚本与预训练 checkpoint 下载链接已发布；不是纯 README 占位。
- **主模型配置：** `detection/tools/cfgs/det_model_cfgs/dsert/ours.yaml`，模型名 `CenterPointMM`，`SECOND_STAGE: True`；LiDAR/4D radar backbone + RGB/thermal/event image backbone。
- **事件预处理源码：** `detection/tools/preprocess_event_voxel.py` 将 `rectified_EVENT_L/*.npz` 的 event(x, y, polarity, timestamp) 转为 5 bins voxel grid（默认 1152×704），生成 `VOXEL_L/*.npz`；需在评测前运行。
- **官方评测入口：** `detection/tools/scripts/dist_test.sh` 用 `torch.distributed.launch` 调用 `detection/tools/test.py`；后者按配置 `build_dataloader`（`dsert_dataset.py`）、`build_network`，载入 checkpoint，再通过 `detection/tools/eval_utils.py::eval_one_epoch` 执行 `model(batch_dict)`、预测生成与 `dataset.evaluation`。
- **复现命令：** 先将数据链接到 `detection/data/dsert-roll`，运行事件预处理并获取 `checkpoint_epoch_20.pth`，在 `detection/tools` 目录执行 `bash scripts/dist_test.sh 0,1 2 --cfg_file cfgs/det_model_cfgs/dsert/ours.yaml --batch_size 4 --ckpt ../checkpoints/checkpoint_epoch_20.pth --extra_tag eval`。
- **依赖边界：** README 测试环境 Ubuntu 20.04/22.04、CUDA 11.1、Python 3.6、PyTorch 1.10、spconv/cu111，存在老版本依赖兼容成本。未在本次环境中实际执行 GPU 评测。
- **许可：** 论文补充材料 §1 标为研究用 CC BY-NC 4.0；须再次核实仓库及依赖许可证。
- **相关：** [DSERT-RoLL 数据集](./dsert-roll-dataset.md) · [官方项目页](../sites/dsert-roll-project.md) · [主实体](../../wiki/entities/paper-dsert-roll.md)
