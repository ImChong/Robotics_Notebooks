# The EuRoC Micro Aerial Vehicle Datasets

> 来源归档

- **标题：** The EuRoC Micro Aerial Vehicle Datasets
- **类型：** paper + dataset
- **出处：** IJRR 2016（online 2015）· IJRR Test of Time Award 2026
- **DOI：** <https://doi.org/10.1177/0278364915620033>
- **数据集页：** <https://ethz-asl.github.io/datasets/>（ASL 主站）
- **作者：** Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas Schneider, Joern Rehder, Samya Omari, Markus W. Achtelik, Roland Siegwart（ETH ASL）
- **入库日期：** 2026-07-20
- **一句话说明：** ETH ASL 发布的 MAV 视觉惯性基准数据集；11 序列同步双目 + IMU + 高精度真值（Leica/Vicon）；工业厂房 + 动捕实验室双场景；十年 VIO/SLAM 标准基准；2026 年 IJRR Test of Time Award。

---

## 核心摘录（策展，非全文）

### 数据集设计动机

- 早期 VIO/SLAM 基准缺乏精密同步真值，难以公平比较不同算法。
- 本数据集目标：提供 **硬同步** 双目相机 + IMU 数据 + **高精度绝对真值**，覆盖不同运动难度，支持算法公平横向比较。

### 采集设置

| 要素 | 规格 |
|------|------|
| 平台 | AscTec Firefly 六旋翼 MAV |
| 相机 | 双目 Aptina MT9V034（全局快门，20Hz，752×480） |
| IMU | ADIS16448（200Hz，六轴） |
| 同步 | 硬触发（相机与 IMU 硬同步） |
| 真值（工厂） | Leica 全站仪，mm 级绝对坐标 |
| 真值（实验室） | Vicon 动捕系统，mm 级 |

### 序列列表（11 个）

- **MH_01_easy ~ MH_05_hard**：工业厂房（Machine Hall），大空间，Leica 真值
- **V1_01_easy ~ V2_03_hard**：Vicon 室，动捕真值

### 评测标准

- ATE（全局位置 RMSE）+ RTE（局部相对误差）；旋转误差辅助
- 与算法无关；多框架可在相同 GT 上比较

### 历史影响

- VINS-Mono（2018）、ORBSLAM3（2021）、Kimera（2019）、Basalt 等众多 VIO/SLAM 旗舰工作均在 EuRoC 上评测。
- Google Scholar 引用数千次，截至 2026 年仍是 VIO 论文默认基准。
- IJRR Test of Time Award 2026 肯定其十年持续影响力。

### 局限

- 室内受控场景，光线稳定，无动态物体；2016 年传感器规格偏低；11 序列总量有限。

### 对 wiki 的映射

- [euroc-mav-datasets](../../wiki/entities/euroc-mav-datasets.md)
- [ego-planner-swarm](../../wiki/entities/ego-planner-swarm.md)
- [orb-slam3](../../wiki/entities/orb-slam3.md)

## 参考来源（原始）

- 论文 DOI：<https://doi.org/10.1177/0278364915620033>
- 数据集：<https://ethz-asl.github.io/datasets/>
