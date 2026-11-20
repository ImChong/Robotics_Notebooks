# leggedrobotics/roam — ROAM 语义导航代码

- **类型：** repo（论文配套代码，不单独建立第二个 wiki 实体）
- **GitHub：** <https://github.com/leggedrobotics/roam>
- **项目：** <https://roam-hiking.github.io/>（[项目归档](../sites/roam-hiking.md)）
- **论文：** <https://arxiv.org/abs/2611.25215>（[论文归档](../papers/roam_legged_navigation_arxiv_2611_25215.md)）
- **许可：** 仓库 README 标注 MIT
- **入库日期：** 2026-11-20
- **核对范围：** 直接读取 README、`vlm_nav/example.py`、`vlm_nav_ros/launch/vlm_nav_ros.launch.py`、`vlm_nav_ros/config/vlm_nav_cfg.yaml`；未运行 GPU 推理/ROS 节点。
- **通读：** 已完成

## 公开模块与入口

| 路径 | 公开范围 / 运行入口 |
|---|---|
| `vlm_nav/` | VLM prompt 与短期记忆、OpenAI/Gemini 客户端、SAN、label→value maps、图像空间 A* |
| `vlm_nav/example.py` | `python3 example.py example_data/hiking_lake.jpg --seg_model api`：`VLMNavAgent.analyze()` → `segment()` → 各通道图像 |
| `vlm_nav_ros/` | `ros2 launch vlm_nav_ros vlm_nav_ros.launch.py`：ROS 2 订阅相机图像/camera info，发布 `_sem` / `_sem_viz`、`~/labels`、`~/response`、`~/duration` |
| `vlm_nav_ros/launch/vlm_nav_elevation_mapping.launch.py` | 与 `elevation_mapping_cupy` 对接启动语义高程图 |
| `vlm_nav_ros1/` | ROS 1 Noetic 对应接口 |
| `examples/elevation_mapping/` | Seealpsee 30 秒语义/传感器数据回放融合示例；需 CUDA GPU |
| `vlm_nav/click_plan_demo.py` | 像素空间目标点击与 A* 演示，**不是**真机 RL 导航器 |

## 工程边界

- README 的 ROS 2 配置 `vlm_duration: 10.0` 与 `segment_duration: 0.5` 是当前仓库可调整参数；不要将其当作论文所有硬件测试通用设置。
- 开源仓覆盖论文架构的语义前端和高程图融合示范；**并未提供从相机到 ANYmal 电机的全系统部署、RL 控制策略训练与权重的完整独立复现路径**。
- 论文最初野外实验 `gpt-4-vision-preview`，当前 README 默认 `gemini-4-hydrogen`；复现实验时必须记录模型版本、API 延迟和分割端设置。

## 对 wiki 的映射

- [paper-roam-open-world-legged-navigation](../../wiki/entities/paper-roam-open-world-legged-navigation.md)
