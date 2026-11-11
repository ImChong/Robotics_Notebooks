# Kimodo-Policy: Learning Kinematic Action Models for Humanoid Loco-Manipulation

- **类型：** 作者一手论文
- **作者：** Yunheng Wang et al.
- **会议：** arXiv 2026 preprint
- **机构：** 香港科技大学（广州）（HKUST Guangzhou）；京东探索研究院（JD Explore Academy）；北京工业大学（BJUT）；剑桥大学（Cambridge）
- **论文：** https://arxiv.org/abs/2611.11554v1
- **原始版本：** arXiv:2611.11554（以所列论文链接版本为准）
- **项目页：** https://kimodo-policy.github.io/
- **代码：** https://github.com/kimodo-policy/kimodo-policy
- **核对范围：** 核对论文官方页面/HTML 可见的方法与结果、作者项目主页；并核对仓库 README 与部分运行入口；全文细节、所有附录未逐段完成交叉核对。
- **通读：** 未完成

## 方法与应用

KAM 的预训练对象是运动学序列，而不是视觉语义；冻结的 Kimodo Root/Body 去噪骨干提供肢体协同先验。Trainable ControlNet 经双路径视觉融合注入根部与身体阶段，HandNet 根据未来身体、图像和当前双手状态生成开合命令。每次使用机器人实测动作历史而非仅重放预定动作。

## 实验与结论

论文报告 HumanoidArena 七任务联合训练平均 72.14%；模型 678M 参数、92M 可训练；真机八任务 60%（48/80），对照移除 419 小时交互数据预训练为 36/80。仿真三项 HOI 消融 18.33/31.67/51.67% 不能和七任务总分混算。

## 局限 / 对 wiki 的映射

文中 1.67 秒预测、0.5 秒执行前缀只属于本策略配置；部署依赖视觉、机器人状态与 SONIC，预训练消融与真实交互数据来源不可混算。

- [Kimodo-Policy：运动动作模型的视觉闭环移动操作](../../wiki/entities/paper-kimodo-policy.md)
