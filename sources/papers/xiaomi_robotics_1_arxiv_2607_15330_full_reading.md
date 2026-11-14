# Xiaomi-Robotics-1 原文核对 / Scaling VLA Models with over 100K Hours of Real-World Trajectories

- **类型：** paper
- **原文：** https://arxiv.org/html/2607.15330v2
- **项目：** https://robotics.xiaomi.com/xiaomi-robotics-1.html
- **代码：** https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
- **机构：** 小米机器人实验室
- **核对范围：** 论文 HTML §1–5（模型/数据两阶段、自动语义标注、数据与模型 scaling、OOD 真机、微调与基准）及此前本仓库的代码/官网归档。
- **通读：** 已完成

## 核心机制与验证口径

- 前训练使用 **100,000+ 小时 UMI 真实世界操纵轨迹**，自动将长轨迹切片、用视觉语言模型为夹爪/物体状态变化标注语言条件；后训练再对齐机器人平台与指令。
- 作者研究规模规律与跨环境泛化，整体因果依赖模型规模、数据量、标注质量、平台差异等多个变量；切勿把“预训练 100k h”写成某一机器人有 100k h 的遥操作数据。
- 真实移动操作和仿真 benchmark 使用不同控制器、数据和评价协议；不能横比其他论文单条最好成功率。
- 工程复现需要检查动作槽位、末端/关节表示、状态归一化、相机标定和安全边界。

## 对 wiki 的映射

- [Xiaomi-Robotics-1 唯一详情](../../wiki/entities/xiaomi-robotics-1.md)
