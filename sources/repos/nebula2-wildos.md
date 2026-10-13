# WildOS（nasa-jpl/nebula2-wildos）

> 来源归档（repo）

- **标题：** WildOS: Open-Vocabulary Object Search in the Wild
- **类型：** repo / ROS 2 / open-vocabulary navigation / outdoor robotics
- **链接：** <https://github.com/nasa-jpl/nebula2-wildos>
- **代码：** <https://github.com/nasa-jpl/nebula2-wildos>
- **论文：** <https://arxiv.org/abs/2602.19308>（见 [wildos_arxiv_2602_19308.md](../papers/wildos_arxiv_2602_19308.md)）
- **项目页：** <https://leggedrobotics.github.io/wildos/>（见 [wildos-project.md](../sites/wildos-project.md)）
- **数据集：** <https://huggingface.co/datasets/leggedrobotics/wildos>（见 [wildos-frontiers.md](../datasets/wildos-frontiers.md)）
- **许可证：** Apache-2.0
- **运行环境：** ROS 2 Jazzy、Python 3.10+、uv、CUDA GPU；README 称实地实验使用 Docker
- **硬件线索：** README 报告使用 RTX 4090 训练、Jetson AGX Orin GPU 部署；实地系统运行在 Boston Dynamics Spot
- **核查日期：** 2026-10-13
- **沉淀到 wiki：** [WildOS 实体页](../../wiki/entities/paper-wildos-open-vocabulary-object-search.md)

## 仓库结构

| 路径 | 作用 |
|------|------|
| nvidia_radio/ | 改造的 RADIO 主干，结合 NACLIP 与 SigLIP2 对齐 |
| explorfm/ | ExploRFM 推理：视觉 frontier、可通行性、开放词汇目标相似度 |
| explorfm_trainer/ | 基于 Lightning + Hydra 的视觉头部训练 |
| visual_navigation/ | ROS 2 WildOS 主流程、基线、目标评分与可视化 |
| triangulation3d/ | 粒子滤波式 3D 目标定位 |
| graphnav_mapper/ | C++ 几何导航图构建 |
| graphnav_planner/ | C++ 图搜索规划 |
| graphnav_msgs/、object_search_msgs/ | 导航图与目标搜索消息定义 |
| ckpts/ | frontier 与 traversability 预测头检查点 |

## 安装与运行入口

- 创建 uv 环境并安装 requirements.txt，再 editable 安装 nvidia_radio/ 和 explorfm/。
- 在 ROS 2 colcon 工作区构建图消息、图建图/规划、三角定位和 visual_navigation 等包。
- 完整目标搜索入口：ros2 launch visual_navigation wildos_launch.py ns:=spot1 do_object_search:=true。
- 另启动 graphnav_planner 与 graphnav_mapper；README 也提供独立三角定位和视觉输出调试入口。
- 外部依赖包括 Elevation Mapping CuPy、DLIO 和 Nav2；启动时需匹配 ROS topic、namespace 与 checkpoint 路径。

## 开放范围与复现边界

仓库提供模型训练/推理代码、ROS 2 系统、C++ 图构建与规划器，以及部分头部权重。C-RADIOv3-B 主干和 SigLIP2 adaptor 需从上游下载。README 指出论文实地实验在 Docker 内运行。完整复现实地结果仍需相应机器人、里程计、建图和局部规划依赖。
