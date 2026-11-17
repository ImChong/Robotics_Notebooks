# FORGE 官方项目页：力感知多粒度人形移动操作

> 项目一手资料归档（2026-11-17 核对）

- **类型：** site / 论文项目主页
- **项目：** FORGE — Force-Aware Multi-Granularity Control for Efficient Humanoid Loco-Manipulation
- **主页：** https://bb0928.github.io/forge/
- **论文：** https://arxiv.org/abs/2611.17142
- **代码：** https://github.com/bb0928/FORGE
- **作者机构：** 上海人工智能实验室、清华大学、浙江大学、上海交通大学、香港中文大学
- **核对范围：** 官方主页的 Research、Controller Design、Controller Comparison、Unified-Control Stability、Contact & Force、Hardware Recovery、Downstream Learning、Resources 区。
- **通读：** 已完成
- **入库日期：** 2026-11-17

## 官方资料给出的能力边界

1. 共用一个全身策略接受导航速度、平面足端、双腕位姿和可选交互力命令；不要求整个人体动作轨迹作为训练监督。
2. 采用冻结行走先验 + 下肢修正 / 上肢动作的非对称残差，结合动量观测器与特权力监督训练的学习补偿器。
3. 官方项目页的 Table I 将**真机**目标响应延迟和跟踪 MPE，与**仿真**双腕夹持反作用力并列显示；这不是一个统一的真机力控基准。
4. Table IV 是代表性单次阶梯负载曲线的 RMSE / MAE，不是所有十次重复的总体均值；35/40/50 N 外推载荷误差更大。
5. Diffusion Policy→FORGE 的箱体蹲起搬运实验共 320 条跨载荷示范；四个负载条件各 20 次真机测试，分别成功 20/20、20/20、18/20、18/20。

## 源码与资源状态

- **部分开源**：项目页的 GitHub 链接可访问；仓库现有残差策略第二阶段训练、Isaac Gym 仿真评估入口和预训练基础模型文件。
- **尚未开源**：官方 README TODO 列出基础策略训练、MuJoCo 部署和真机部署脚本。
- **没有承诺**可依据现有代码独立复现官方全部真机与接触实验；具体入口归档见 [FORGE 官方代码](../repos/forge.md)。

## 对 wiki 的映射

- [FORGE 论文唯一实体节点](../../wiki/entities/paper-forge-force-aware-loco-manipulation.md)
- [论文原文归档](../papers/forge-force-aware-loco-manipulation_arxiv_2611_17142.md)
