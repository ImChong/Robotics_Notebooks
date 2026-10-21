# RTISSLAM: In-air Imaging Sonar SLAM using a Robust Pose Graph and Occupancy Grid Mapping（arXiv:2610.18479）

> 来源归档（ingest）

- **标题：** RTISSLAM: In-air Imaging Sonar SLAM using a Robust Pose Graph and Occupancy Grid Mapping
- **arXiv：** <https://arxiv.org/abs/2610.18479>（cs.RO，2026-10-13 提交，CC BY-NC-ND 4.0）
- **代码：** <https://github.com/Cosys-Lab/RTISSLAM>
- **项目页：** 无独立项目页（论文提及演示视频 <https://youtu.be/OV5ceQ0kd-c>）
- **作者：** Jan Steckel（Cosys-Lab，安特卫普大学应用工程学院；Flanders Make 战略研究中心）
- **入库日期：** 2026-10-17
- **来源线索：** 具身智能小站 2026-10-17 文章（<https://mp.weixin.qq.com/s/I4XP5JSG9QNmm4hd1nk2JQ>）
- **一句话说明：** 纯声呐 SLAM：鲁棒位姿图 + 占据栅格，回环后地图更新，结合声学里程计标定轮式里程计。

## 开源状态（步骤 2.5）

- **无项目页**；论文 HTML 给出源码地址。
- **仓库核查（2026-10-17）：** BSD-3-Clause；目录 `rtissim/`、`rtisslam/`、`rtisnav/`、`batslam/`、`sondec/`、`ros/`（ROS 2 Jazzy）、`scripts/`、`tools/`、`tests/`；README 含安装、冒烟测试、demo 与论文复现命令；仅 1 次提交，引用信息"待补"。
- **结论：** **已开源**。本次未在本地运行。详见 [仓库归档](../repos/rtisslam.md)。

## 摘录 1：方法

- 能量图：DMAS3 + CF 波束形成 + RW-CFAR；局部视图（20 方位扇区）与每 0.3 m 模板做位置识别，BatSLAM 2.0 序列验证；回访经相关扫描匹配做度量配准；声学里程计对轮式里程计做最小二乘尺度标定；iSAM2 + Huber 核。
- 占据栅格以关键帧存证据，随位姿图更新；栅格不反馈位姿。

**对 wiki 的映射：** [`paper-rtisslam`](../../wiki/entities/paper-rtisslam.md)「核心原理」。

## 摘录 2：仿真与结果

- 三个镜面墙世界：办公室 36×22 m、仓库 32×26 m（L 形）、中庭（八边形半径 12 m）；轮式里程计 2% 尺度误差与 3% 航向偏差；5 种子。
- ATE（m）办公室/仓库/中庭：轮式 8.20/5.09/2.66；RTISSLAM 0.31/0.57/1.78；slam_toolbox 默认 23.1/35.5/17.3。
- 位置识别：34%（BatSLAM 2.0 设置）→ 54%（调参），95% 精度，AUC 0.97。

**对 wiki 的映射：** 同页「实验与评测」。

## 摘录 3：局限（作者自述）

- 仅仿真，真实评测与激光参考为后续工作；中庭里标定假设失败；常量轮偏差假设；栅格不约束位姿。

**对 wiki 的映射：** 同页「局限与风险」。

## 交叉链接

- 仓库归档：[rtisslam.md](../repos/rtisslam.md)
