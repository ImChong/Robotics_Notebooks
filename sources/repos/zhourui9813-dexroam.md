# zhourui9813/DexRoam

> DexRoam 人体采集 / 人机动作对齐源码归档

- **类型：** repo
- **链接：** <https://github.com/zhourui9813/DexRoam>
- **项目页：** <https://dexroam.github.io/>
- **论文：** <https://arxiv.org/abs/2609.35761v1>
- **许可证：** MIT（仓库 LICENSE）
- **入库日期：** 2026-10-04
- **源码核对：** 2026-11-11，main 分支（`git/trees/main` SHA `e539387a847ac3f70285bfd9ceb35cfff98db614`）
- **一句话说明：** 从 Quest 3 + ZED Mini 采集头部/身体/手部及双目图像，存入 HDF5；再将人体全身轨迹重定向到 Astribot / XHand 并按机器人 episode 长度重采样。

## 实际可执行入口

| 官方入口 / 模块 | 核对到的职责 |
|---|---|
| `python -m ego_wholebody_mocap` | `__main__.py → main.main()`；`VuerRecorder.step()` 获取人体姿态，`ZedCameraSession` 获取双目帧，`GestureCaptureController` 控制 episode 状态 |
| `ego_wholebody_mocap/storage/episode_manager.py` | `EpisodeCaptureManager` 启停、保存与丢弃 episode；经 `AsyncEpisodeWriteBroker` 异步写入 HDF5 |
| `python -m dexroam_alignment input -o output --recursive --robot-dir robot_dir` | `__main__.py → cli.main() → runner.retarget_path()` |
| `runner.py` | `build_retarget_frames()` 读 HDF5，`pipeline.retarget_sequence()` 按帧处理，`write_output()` 保存重定向结果 |
| `pipeline.py` / `geometry.py` / `hands.py` | 骨盆局部根系、底盘 XY-Yaw、上身/头部几何映射与 XHand 关节重定向 |
| `resampling.py` | `compute_robot_target_length()` 求机器人 episode 均长；`resample_directory()` 和 `resample_hdf5_file()` 重采样数据 |

## 复现入口（按官方 README）

```bash
# Quest3+ZED 硬件与依赖准备完成后
python -m ego_wholebody_mocap

# 独立 Python 环境（官方建议 Python 3.10）
pip install -r human_robot_alignment/requirements_alignment.txt
cd human_robot_alignment
python -m dexroam_alignment /path/to/human_dir -o /path/to/output_dir \
  --recursive --robot-dir /path/to/robot_dir
```

`--robot-dir` **只用于计算目标 episode 平均长度，不参与重定向的姿态求解**；也可以改用 `--target-length` 或 `--speed-factor`，三者互斥。源码实际保存流程涉及临时 HDF5 和二次重采样，不应画作一个实时的端到端同步函数调用。

## 开源边界

**采集与对齐代码已开源**；不是面向任意机器人即装即用的通用策略部署包。需要 Meta Quest 3、ZED Mini、其 SDK、WebXR/VR 配置、对应 XHand URDF/配置与实际数据。策略训练独立在 [DexRoam-Policy-Training](./zhourui9813-dexroam-policy-training.md)。未在本环境执行硬件录制或真机重放。

## 对 wiki 的映射

- [DexRoam 论文主实体与源码时序图](../../wiki/entities/paper-dexroam-mobile-bimanual-manipulation.md)
- [项目页归档](../sites/dexroam-github-io.md)
- [论文全文归档](../papers/dexroam_arxiv_2609_35761.md)
