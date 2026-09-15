# HumanPlus-1000（Hugging Face）

> 来源归档（ingest 配套数据集）

- **标题：** HumanPlus-1000
- **类型：** dataset / huggingface
- **链接：** <https://huggingface.co/datasets/humanplus-ai/humanplus-1000>
- **项目页：** <https://humanplus-ai.github.io/HumanPlus1000.github.io/>
- **代码 / 可视化：** <https://github.com/humanplus-ai/humanplus-1000>
- **机构：** HumanPlus
- **许可：** CC BY-NC 4.0
- **访问：** 预览子集 **ungated**；全量联系 info@humanplus.xyz
- **入库日期：** 2026-09-15
- **一句话说明：** 大规模同步 egocentric 视觉 + 全身 SMPL-H + MANO 手 + SLAM/IMU/深度 的真实世界人类行为数据；HF 当前为 **100 session 预览**（约 **71.8 GB**），非最终 1000h 分布。

---

## 规模（HF / 项目页，入库日快照）

| 指标 | 数值 |
|------|------|
| 全量目标 | **1000+** 小时（项目页） |
| 预览发布 | **100** demo sessions（HF README） |
| Hub 体量 | 约 **71.8 GB**（HF 侧栏） |
| 采集规模（项目页） | **200+** 人 · **500+** 任务 · **100+** 地点 |
| 活动标签 | Viewer 展示 **33** 类（预览子集） |
| 许可 | CC BY-NC 4.0 |

> 预览子集用于展示格式、模态、标注与用例；**不代表** HumanPlus-1000 最终规模与完整分布。

## Session 目录结构

```text
HumanPlus-1000/
├── metadata/
│   └── sessions.parquet
└── data/
    ├── HP_S000001/
    │   ├── fisheye_left.mp4
    │   ├── fisheye_right.mp4
    │   ├── annotation.hdf5
    │   └── metadata.json
    └── ...
```

HF 预览路径示例：`data/session_20260903_113804/`（含同名四文件）。

## `annotation.hdf5` 顶层组（README）

| 组 | 内容 |
|----|------|
| `calibration/` | 鱼眼/立体内参畸变、校正 stereo、头–相机、mocapworld↔slamworld、body IMU |
| `video/` | 左右鱼眼文件名、分辨率、`timestamp_ns` |
| `slam/` | `T_slamworld_camera`、点云 |
| `depth/` | 深度图、内参、有效范围 |
| `imu/` | 头/身 IMU 加速度、角速度、姿态 |
| `body_motion/` | `T_mocapworld_root`、`smplh_pose`、`body_keypoints`、脚接触概率 |
| `hand_motion/` | 左右 MANO 姿态、21 关节、腕部相机坐标、valid/confidence |
| `synchronization/` | UTC 与视频帧索引对齐 |
| `behavior_annotation/` | 活动摘要、运动叙述、原子动作 |

## 模态清单

- **Egocentric vision：** 立体鱼眼、校正视图、深度
- **Human motion：** SMPL/SMPL-H、全局根位姿、脚接触
- **Hand motion：** MANO 参数与 3D 关节
- **World motion：** SLAM 轨迹、相机在重建世界中的位姿
- **Inertial：** 可穿戴 IMU（头/身）
- **Semantic metadata：** 场景、任务/活动、时长、模态可用性、质量指标

## 快速用法

```bash
# 下载后
python examples/example_load_annotation.py --data_root /path/to/session
python -m visualize --session-dir /path/to/session
```

官方 loader 见 [humanplus-ai/humanplus-1000](https://github.com/humanplus-ai/humanplus-1000)。

## 对 wiki 的映射

- [humanplus-1000-dataset](../../wiki/entities/humanplus-1000-dataset.md)
- [humanplus-1000.md](../sites/humanplus-1000.md)
- [humanplus-1000.md](../repos/humanplus-1000.md)
- [paper-loco-manip-161-012-humanplus](../../wiki/entities/paper-loco-manip-161-012-humanplus.md) — 同品牌人形 shadowing 方法对照
