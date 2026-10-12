# Human-Like Redundancy Resolution for Real-Time Bimanual Humanoid Teleoperation

> 来源归档（ingest）

- **类型：** paper
- **论文：** <https://github.com/chaotan2003/human-like-redundancy/blob/main/paper.pdf>
- **项目页：** <https://chaotan2003.github.io/human-like-redundancy/>
- **代码：** <https://github.com/IAIE-PhysicalAI/Bimanual-Teleoperation>（原仓链接重定向）
- **作者：** Chao Tan、Ming Shin Cindy Sung、Shuxian Liang（通讯）、Chaoqun Wan、Zhen Huang、Xian-Sheng Hua
- **机构：** Tongji University
- **入库日期：** 2026-10-12
- **一句话说明：** 以肩—腕几何映射、自适应肘偏好和层级速度 IK 零空间解析，改善双臂遥操作的类人冗余姿态。
- **沉淀到 wiki：** [实体页](../../wiki/entities/paper-human-like-redundancy-resolution.md)

## 方法与结果摘录

SWGR 从人体肩—腕几何构造机器人尺度的腕目标；AEAC 使用肘偏移方向引导姿态，并在人臂伸直时减弱或停用；J-PARSE 将肘偏好放入腕部任务零空间，腕跟踪优先。固定腕目标消融中肘方向平均误差下降 54.4%；工作空间边界压力测试中峰值原始关节速度需求降低 97.9%；静态腕位置误差均值 2.87 mm、手臂平面角误差 2.42°。以上属于不同评测指标，不是完整任务成功率。

## 公开实现范围

项目页链接到转移后的 IAIE-PhysicalAI/Bimanual-Teleoperation。README 的脚本覆盖 PICO 输入、JAKA K1 目标重定向、J-PARSE IK 和 Viser 可视化；未列物理仿真后端或真实硬件输出链路。详见[仓库归档](../repos/bimanual-teleoperation.md)。

## 对 wiki 的映射

- [实体页](../../wiki/entities/paper-human-like-redundancy-resolution.md)
- [Teleoperation](../../wiki/tasks/teleoperation.md)
- [Inverse Kinematics](../../wiki/formalizations/inverse-kinematics.md)
