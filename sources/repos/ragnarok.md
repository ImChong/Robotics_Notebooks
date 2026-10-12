# RAGNAROK 代码仓库

> 来源归档

- **标题：** hanjun815/RAGNAROK
- **类型：** repo
- **链接：** <https://github.com/hanjun815/RAGNAROK>
- **论文：** <https://arxiv.org/abs/2610.11531>
- **项目页：** <https://ragnarok-rvki-slam.github.io/RAGNAROK/>
- **许可：** BSD 3-Clause（根目录 LICENSE；`src/Exteroceptive` 继承 OKVIS2-X 许可）
- **入库日期：** 2026-10-12（clone 的 HEAD `99e9c371`，提交日期 2026-10-09）
- **一句话说明：** 腿式机器人雷达–视觉–运动学–惯性 SLAM 的官方实现：ROS 2 Humble + CUDA 12.1 Docker 环境，含本体（Proprioceptive）与外部感知（Exteroceptive，基于 OKVIS2-X）两部分。
- **沉淀到 wiki：** [`wiki/entities/paper-ragnarok-radar-visual-kinematic-inertial-slam.md`](../../wiki/entities/paper-ragnarok-radar-visual-kinematic-inertial-slam.md)

## 目录结构

| 路径 | 说明 |
|------|------|
| `docker/build.sh`、`docker/run.sh` | 构建 `ragnarok:humble-cu121` 镜像；挂载数据集根目录并转发 GPU/X11 |
| `build_workspace.sh`、`setup_env.sh` | 容器内按依赖顺序编译 tiny-viewer、CTraj、`spot_msgs`、`garlileo`、`okvis` |
| `src/Proprioceptive/` | 雷达辅助本体估计器（`exe`、`launch`、`config/<dataset>`），B 样条 + 腿速/雷达因子 |
| `src/Exteroceptive/` | OKVIS2-X 派生的视觉惯性 SLAM 后端（`okvis_frontend`、`okvis_ceres`、`okvis_mapping`、`supereight2`、`okvis_ros2` 等） |
| `src/spot_msgs/` | Spot 腿运动学消息定义 |

## 运行入口（README）

- `ros2 launch okvis ragnarok.launch.py config_filename:=…/okvis2_SLAM.yaml se_config_filename:=…/se2.yaml config_path:=…/Proprioceptive/config/RAGNAROK/config.yaml rosbag_path:=<序列>/ros2_bag`：RAGNAROK-SLAM（含回环与最终 BA）。
- 同命令换 `okvis2.yaml`：RAGNAROK-Odom（无回环）。
- `ragnarok_RVI.launch.py` / `ragnarok_VKI.launch.py`：去掉腿 / 去掉雷达的变体；`garlileo garlileo.launch.py`：仅雷达 + 腿的 RKI。
- 结束：`ros2 service call /okvis/shutdown std_srvs/srv/SetBool "{data: true}"`，轨迹写入 `results/`，本体轨迹为 `results/proprioceptive/poses.txt`。
- 数据：RAGNAROK 自采数据集、GaRLILEO、Co-RaL 均通过 README 中的链接/表单获取；后两者无相机，只能跑 RKI。
