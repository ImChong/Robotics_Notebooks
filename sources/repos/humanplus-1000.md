# humanplus-ai/humanplus-1000 — 原始资料归档

- **来源：** <https://github.com/humanplus-ai/humanplus-1000>
- **类型：** repo / dataset tooling
- **项目页：** <https://humanplus-ai.github.io/HumanPlus1000.github.io/>
- **数据集：** <https://huggingface.co/datasets/humanplus-ai/humanplus-1000>
- **机构：** HumanPlus
- **归档日期：** 2026-09-15
- **默认分支：** `main`

## 一句话说明

**HumanPlus-1000 Viewer**：读取与可视化 HF 上 `annotation.hdf5` 会话的官方工具链——加载标定/SLAM/全身与手部运动/深度/点云，并用 [Rerun](https://rerun.io) 展示立体鱼眼、校正视图、深度、SMPL-H 网格与世界坐标 SLAM 轨迹。

## 开源核查（步骤 2.5）

| 项 | 结论（截至 2026-09-15） |
|----|-------------------------|
| **代码** | **已开源** — `data_loader.py`、`visualize.py`、`geometry.py`、`body_model.py` 等 |
| **数据** | 不在仓内；从 HF 下载 session 目录 |
| **SMPL-H 网格** | 需官方 `model.npz`（`--smplh-model` 或 `HUMANPLUS_SMPLH_MODEL`）；缺省回退 SMPL-24 骨架 |
| **许可证** | 查看器 **MIT**；数据 **CC BY-NC 4.0**（见 README） |

## 目录与入口（README）

| 路径 | 作用 |
|------|------|
| `data_loader.py` | 加载 `annotation.hdf5` 与立体视频 |
| `visualize.py` | `python -m visualize --session-dir …`；可选 `--output-rrd` |
| `geometry.py` | 深度 colormap、点云、去畸变、骨架辅助 |
| `blueprint.py` | Rerun 布局 |
| `body_model.py` | NumPy SMPL-H LBS（可选 mesh） |
| `examples/example_load_annotation.py` | 列出 HDF5 内容并打印摘要 |

## 安装与快速开始

```bash
conda create -n humanplus python=3.11
conda activate humanplus
pip install -r requirements.txt
pip install -e .
python examples/example_load_annotation.py --data_root /path/to/session
python -m visualize --session-dir /path/to/session
```

Session 目录典型结构：

```text
HP_S000001/
  fisheye_left.mp4
  fisheye_right.mp4
  annotation.hdf5
  metadata.json
```

## 坐标系（README）

- 展示：**mocapworld**，Y-up（`ViewCoordinates.RIGHT_HAND_Y_UP`）
- SLAM 点云/深度/相机位姿经 `calibration/T_mocapworld_slamworld` 变换
- 手部 overlay：`joints_cam = joints_3d (MANO local) + wrist_position_camera`，再经 `R_rect_*` 校正

## 对 wiki 的映射

- [humanplus-1000-dataset](../../wiki/entities/humanplus-1000-dataset.md)
- [humanplus-1000.md](../sites/humanplus-1000.md)
- [humanplus-1000.md](../datasets/humanplus-1000.md)
