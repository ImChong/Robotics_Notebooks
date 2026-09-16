# Learning Multi-Agent Task Assignment and Navigation in the Factory: from Simulation to Real Robots

> 来源归档（ingest）

- **标题：** Learning Multi-Agent Task Assignment and Navigation in the Factory: from Simulation to Real Robots
- **简称：** FMAPPO
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.14567>
- **PDF：** <https://arxiv.org/pdf/2609.14567>
- **代码：** 截至入库日 **未见** GitHub
- **项目页：** <https://anonymouspapers123.github.io/FMAPPO/>
- **入库日期：** 2026-09-15
- **索引来源：** [具身智能小站 9+EffVLA 盘点](../blogs/wechat_embodied_station_9_papers_resources_effvla_2026-09-15.md)
- **一句话说明：** 去中心化 2D LiDAR + 任务状态 MAPPO；仿真零件交付 +106%，真机暴露频率/通信 sim-to-real 差。

## 开源状态（步骤 2.5，2026-09-15）

**结论：待发布**

## 核心摘录

### 摘录 1

去中心化 2D LiDAR + 任务状态 MAPPO；仿真零件交付 +106%，真机暴露频率/通信 sim-to-real 差。

**对 wiki 的映射：** [paper-fmappo](../../wiki/entities/paper-fmappo.md)

### 摘录 2（官方 abstract 要点，2026-09-15 补录）

- **论文题名：** *Learning Multi-Agent Task Assignment and Navigation in the Factory: from Simulation to Real Robots*。
- **问题：** MARL 在工业环境的 **物理多机系统** 上落地仍难；本文考察去中心化 MARL 用于 **多机器人多机台看护（multi-machine tending）** 的真实可用性。
- **方法：** **Feature-fusion MAPPO（FMAPPO）** — 融合 **2D LiDAR 测量** 与 **任务相关状态信息**，实现安全的去中心化任务分配与导航。
- **落地管线：** 用高保真机器人仿真 + **ROS2** 搭完整 sim-to-real 管线，部署到 **移动操作平台**（实验中 **机械臂关闭**，只跑移动与任务分配）。
- **额外考察：** 学到的策略对 **指令更新频率** 的敏感性（部署侧重要参数）。
- **仿真对照结果（vs MAPPO / SMAPPO，effect size 大）：**
  - 零件 **交付**：**+106%** / **+21%**；
  - 零件 **收集**：**+48%** / **+11%**；
  - **机台利用率**：**+31** / **+10** 个百分点；
  - **碰撞**：**−18%** / **−15%**；
  - **安全分**：**+14** / **+6** 个百分点。
- **真机结果：** 去中心化策略可协调多机服务多机台，并在真实感知与控制约束下保持安全运行（abstract 未给真机量化表）。
- **视频：** <https://anonymouspapers123.github.io/FMAPPO/>

**对 wiki 的映射：** 同上（补入该页「核心原理（方法）」「实验与评测」「与其他工作对比」三节）

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射
