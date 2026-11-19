# 港科大（广州）：Autane 自主起重机研究介绍（2026-11-16）

> 来源归档（官方机构新闻，非论文全文；不是独立 Autane 项目主页）。

- **类型：** site / 官方研究报道
- **标题：** HKUST(GZ) Autonomous Crane System Makes Science Robotics Cover
- **来源：** 香港科技大学（广州）官方网站
- **链接：** <https://www.hkust-gz.edu.cn/2026/11/16/hkustgz-autonomous-crane-system-makes-science-robotics-cover/>
- **论文：** <https://www.science.org/doi/10.1126/scirobotics.ady5784>
- **入库日期：** 2026-11-19
- **核对范围：** 2026-11-16 该官方研究新闻正文与指标段落，未核验论文补充材料。
- **通读：** 已完成

## 官方报道摘录

- **部署：** 在上海、宁波、苏州、香港等港口的 60 多台起重机上，适配 5 类不同代际机型；报道给出累计超过 480 万 TEU 的处理量。
- **感知与决策：** LiDAR、相机、GNSS、IMU + 融合/SLAM/障碍检测 + 规划/控制/运维闭环。
- **负载控制：** 报道点名 MPC 和扰动抑制以降低集装箱悬吊摆动，并提及传感器自动重标定。
- **评测披露：** 超过 6 万次作业的平均成功率 91.23%、效率 25.93 箱次/小时；约 90% 瞬态故障可自主消除。
- **口径注意：** 该报道与期刊摘要所写“超过 200 万 TEU”不一致，二者可能对应不同统计日期或范围，未获得原始数据前不作归因。

## 开源状态

该官方报道未列出专属项目主页、代码仓库、权重或数据集下载入口；本次未核实到 Autane 可运行源码，**不能宣称已开源**。无源码运行时序图。

## 对 wiki 的映射

- [Autane 论文实体](../../wiki/entities/paper-autane-autonomous-crane.md)
- [对应论文归档](../papers/autane_science_robotics_ady5784.md)
