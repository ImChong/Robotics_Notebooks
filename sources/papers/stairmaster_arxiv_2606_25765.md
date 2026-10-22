# StairMaster 论文来源归档

## 元数据

- **题目：** StairMaster: Learning to Conquer Risky Hollow Stairs for Agile Quadrupedal Robots
- **作者：** Xincheng Tang, Youhan Xie, Zhengjie Shu, Wanyu Li, Lai Jiang, Wenkang Hu, Yitong Li, Ruigang Yang
- **单位：** Shanghai Jiao Tong University（上海交通大学）
- **预印本：** https://arxiv.org/abs/2606.25765（首次提交 2026-06-24，v2 2026-07-29）。该 ID 由项目页链接核实，并非来自原始输入。
- **项目主页：** https://sivan666666.github.io/StairMaster/
- **项目页 GitHub：** https://github.com/Sivan666666/StairMaster（仓库内容为学术项目页模板，未包含 StairMaster 的训练或部署代码、权重或数据）
- **入库日期：** 2026-10-22

## 论文要解决的问题

四足机器人攀爬镂空楼梯（hollow stairs）时存在腿部被卡住的风险，且深度感知稀疏、噪声大。

## 方法要点（据 arXiv 摘要）

- 三阶段强化学习框架。
- Cross-Attention：从噪声深度输入中提取结构特征。
- Spatial-aware Recurrent Unit：维持时空记忆，补偿感知盲区。
- sim-to-real 深度传感器模型，缩小感知差距。
- 3D waypoint 引导的主动感知奖励，并对镂空缝隙与楼梯边缘施加惩罚，以改善落脚点选择。

## 报告的结果（据 arXiv 摘要）

- 在 Unitree Go2 上零样本迁移部署。
- 可攀爬坡度最高 55° 的镂空楼梯。
- 作者声称这是首个在真实世界实现此类陡峭镂空楼梯攀爬的 RL 策略。

## Robotics_Notebooks 映射

- 实体页：[StairMaster](../../wiki/entities/paper-stairmaster.md)

## 一手来源

- arXiv 摘要页与项目页（经 WebFetch 摘要）
