# PolicyIK 官方项目页面

> 一手项目页归档；与论文与代码属于同一项目，wiki 只维护一个实体。

- **标题：** PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning
- **中文标题：** PolicyIK：基于强化学习的高精度机械臂逆运动学
- **类型：** site / project
- **链接：** <https://tinaliu0123.github.io/PolicyIK/>
- **论文：** <https://arxiv.org/abs/2611.05372>
- **代码：** <https://github.com/Tinaliu0123/PolicyIK>
- **入库日期：** 2026-11-08（增量核验）
- **一句话说明：** 目标条件化强化学习策略控制机械臂 6-DoF 末端位姿，强调到达目标后的稳定保持。
- **核对范围：** 项目主页公开的论文、Code 入口、可视化演示与性能介绍，结合原论文和 README 复核数字。
- **通读：** 已完成
- **沉淀到 wiki：** [PolicyIK 统一实体](../../wiki/entities/paper-policyik.md)

## 开源核查（2026-11-08）

- **已开源**：项目链接指向官方 GitHub；仓库已有仿真训练、评估、checkpoint、任务数据，和真实 OpenArm 的训练、评测、ROS 2 接入、示范数据等。
- **仿真成功率**：98.9% 对应多次重试的部署协议；单次执行 93.5%，详见 [论文阅读归档](../papers/policyik_arxiv_2611_05372.md)。
- **真机成功率**：OpenArm reached 54/60，held 43/60；另一组 xArm7 实验成功率应按对应场景和测试条件解读。
- **复现边界**：公开代码中可辨认 OpenArm 仿真与真机入口；不能仅因论文报告 xArm7 结果，就推定仓库含同等完整的 xArm7 复现脚本。

## 交叉链接

- [论文一手阅读归档](../papers/policyik_arxiv_2611_05372.md)
- [代码结构与入口](../repos/policyik.md)
