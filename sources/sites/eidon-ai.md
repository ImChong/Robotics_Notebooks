# Eidon AI（归档项目页）

- **标题：** Eidon AI Robotics Data
- **类型：** 项目页 / 可穿戴硬件 / 仿真与可视化 / 数据集
- **项目页：** <https://www.eidon.ai/>
- **GitHub 组织：** <https://github.com/Eidon-AI>
- **Hugging Face 组织：** <https://huggingface.co/eidon-ai>
- **白皮书：** <https://www.eidon.ai/api/whitepaper>
- **状态：** 公司已停止运营；官方归档页仍保留公开代码、硬件设计和数据入口
- **核查日期：** 2026-10-11
- **数据许可：** 数据资源标注 CC-BY-4.0；代码/硬件按各仓库及组件 LICENSE
- **沉淀到 Wiki：** [Eidon Tracker + POV 数据集](../../wiki/entities/eidon-tracker-pov-dataset.md)

## 一句话摘要

Eidon 发布了面向具身学习的人类日常操作采集系统，包括 7-IMU 上肢 Tracker、16-DOF 手套、Eidon Sim 可视化/运动学工具，以及配对 Tracker+POV 与纯 POV 两类数据。

## 官方资源清单

官方归档页声明硬件、仿真环境和数据集已开放。公开入口包括三个 GitHub 仓库及三个 Hugging Face 数据资源：

| 资源 | 内容 | 入口 |
|------|------|------|
| Eidon Tracker | 电池供电 IMU→Bluetooth 穿戴设备，完整 7-IMU 方案见白皮书 | [代码](https://github.com/Eidon-AI/eidon-tracker) · [白皮书](https://www.eidon.ai/api/whitepaper) |
| Eidon Glove | Hall-effect 手指追踪手套，含固件、PCB/CAD 和主机端可视化 | [代码](https://github.com/Eidon-AI/eidon-glove) |
| Eidon Sim | 设备数据可视化、手臂运动学、视频同步轨迹回放；Eidon Sym 的开源后继项目 | [代码](https://github.com/Eidon-AI/eidon-sim) |
| Tracker + POV 视频/元数据 | 13,451 段配对录制，MP4 和元数据 | [Hugging Face](https://huggingface.co/datasets/eidon-ai/tracker-pov) |
| Tracker IMU 表 | 同一批录制的约 779M 行 IMU，按 recording_id 关联 | [Hugging Face](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu) |
| 纯 POV 视频 | 不带 Tracker 的视频；Storage Bucket，不是标准 Dataset 仓库 | [Hugging Face Bucket](https://huggingface.co/buckets/eidon-ai/egocentric-pov) |

配对集为 **1,273.8 小时、13,451 段、27 位贡献者**；纯视频集为 **305.7 小时、1,370 段、37 位贡献者**。官方首页总计 **1,579 小时、14,821 段、10.6 TB**。数据卡称两个子集有 14 位重复贡献者，因此不能直接相加分集人数作为独立人数。

## 系统与开放边界

- **Tracker：** 7 个 BNO085 9-DOF IMU，分布在胸部、上臂、前臂和手部；官方标注 24 Hz、约 1–2° RMS、8–12 小时续航。
- **Glove：** Hall-effect 传感器和嵌入磁体测量手指关节角，约 100 Hz BLE、16 DOF。
- **Eidon Sym / Eidon Sim：** 官网称历史 Eidon Sym 曾运行 VLM 质量控制、自动过滤/标记录制并导出仿真兼容格式；Eidon Sim 是其开源后继项目，提供七自由度手臂运动学、3D 骨架可视化和视频同步回放。不要把后继工具等同于历史在线服务。
- **采集 App：** iOS/Android App 曾负责 BLE 配对、视频和传感器同步采集；官网明确说明其依赖的后端已关闭。归档不表示托管采集服务仍可用。
- **许可证：** Tracker 与 Eidon Sim 标注 MIT；Glove 仓库要求按组件核对许可。数据与代码许可分开判断。

数据大小存在口径差异：官网列 IMU 表约 9.54 GB；Hugging Face 卡片概览写约 9.5 GB，结构表另写约 6.6 GB。实际下载量以当前文件索引为准。

## 来源映射

- [Eidon Tracker + POV 数据集实体页](../../wiki/entities/eidon-tracker-pov-dataset.md)
- [项目仓库归档](../repos/eidon-ai.md)
- [数据卡与字段说明](../datasets/eidon-tracker-pov.md)
