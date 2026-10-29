# Pushing Dexterity and Safety（arXiv:2610.37652）

> 来源归档（论文）

- **标题：** Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well
- **作者：** Ian Chuang、Hiya Shah、Vamsi Chilamkur、Iman Soltani、Claire Tomlin
- **机构：** 加州大学伯克利分校、加州大学戴维斯分校
- **论文：** <https://arxiv.org/abs/2610.37652>
- **项目页：** <https://ian-chuang.github.io/dexsafe/>（[归档](../sites/dexsafe.md)）
- **入库日期：** 2026-10-29

## 核心内容

- 使用生成抓取数据、多样目标与自适应摩擦课程，在仿真中训练 LEAP 手内位姿到达策略，无需人类数据。
- 系统辨识后零样本迁移，采用动作捕捉状态或腕部深度相机；学习式残差安全过滤器调整动作以降低掉落。
- 真机 4 物体、4,800 个未见目标、13.3 小时评测；状态策略在 2 cm / 0.4 rad 阈值下达成 95.2%，过滤器使掉落从 16 次降至 4 次，同时成功率为 91.5%。

## 对 wiki 的映射

- [DexSafe](../../wiki/entities/paper-dexsafe.md)
