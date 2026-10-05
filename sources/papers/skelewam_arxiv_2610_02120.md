# SkeleWAM（arXiv:2610.02120）

> 来源归档（paper）

- **标题：** SkeleWAM: Skeleton World-Action Modeling for Efficient Robotic Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.02120>
- **HTML：** <https://arxiv.org/html/2610.02120v1>
- **PDF：** <https://arxiv.org/pdf/2610.02120>
- **项目页：** <https://skelewam-project.github.io/>
- **机构：** 北京大学
- **作者：** Juyi Sheng、Hua Wang、Mengyuan Liu
- **版本：** v1，2026-10-01
- **许可：** arXiv 页面列示 CC BY-NC-ND 4.0（论文文本许可；不代表代码许可）
- **代码：** 截至 2026-10-05，论文与官方项目页未链接到公开训练/推理代码仓库或权重
- **一句话说明：** 用机器人关节、物体中心和交互点构成稀疏 3D 骨架，把动作生成与未来骨架预测共同训练；推理时只保留动作分支，并用 Medoid Action Consensus 从采样轨迹中选代表轨迹。

## 论文要点

1. **状态表示：** RGB-D 感知物体中心和交互点，配合正向运动学得到机器人关节/末端关键点；统一到机器人中心坐标系，形成身份和连接关系固定的稀疏骨架。
2. **联合目标：** 同时用 flow matching 生成动作块和未来骨架序列。未来几何预测为动作策略提供辅助监督，不要求重建未来图像。
3. **推理：** 去掉未来骨架预测分支，根据当前骨架和语言指令生成动作；Medoid Action Consensus（MAC）选择与其他候选平均距离最小的一条完整轨迹，不需要奖励/价值模型，也不对轨迹求平均。
4. **LIBERO-Plus：** 10,030 个扰动变体、七类扰动的零样本评测；RGB-D 主设置整体成功率 85.9%，57.1M 参数；相较 Cosmos-Policy（82.2%）高 3.7 个百分点。布局扰动成功率为 66.6%，是明显短板。
5. **真机：** ARX R5 + 外部及腕部 RealSense 相机；开抽屉、关抽屉、叠积木、叠碗、把积木放进抽屉五项任务，每方法每项 20 次试验；SkeleWAM 平均成功率 89%。

## 对 wiki 的映射

- [paper-skelewam-efficient-manipulation](../../wiki/entities/paper-skelewam-efficient-manipulation.md)
- [项目页归档](../sites/skelewam-project.md)
- [相近但不同的 SkelWAM（arXiv:2609.21983）](https://arxiv.org/abs/2609.21983)：该论文聚焦跨具身迁移，不能与本论文合并。
