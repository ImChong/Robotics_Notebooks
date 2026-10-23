# Watch Your Step: Anticipated Perception for Humanoid Locomotion

- **类型：** paper
- **作者：** Matthias Heyrman, Chenhao Li, Ioannis Dadiotis, Marco Hutter
- **机构：** ETH Zürich, Robotic Systems Lab；Chenhao Li 另属 ETH AI Center
- **提交日期：** 2026-10-21（arXiv v1）
- **链接：** [arXiv:2610.29612](https://arxiv.org/abs/2610.29612)
- **项目页：** <https://leggedrobotics.github.io/watch-your-step/>
- **代码：** <https://github.com/leggedrobotics/foothold-prediction-model>
- **入库日期：** 2026-10-23
- **一句话说明：** 通过预测下一步足点，只从高程图提取相关局部区域，减少人形机器人感知式行走策略的输入冗余。
- **沉淀到 wiki：** [Watch Your Step](../../wiki/entities/paper-watch-your-step.md)

## 核心摘录

1. **问题：** 感知式运动策略常直接输入稠密的机器人中心高程图，但策略决策主要受即将落脚位置附近的地形影响；让策略自行从全图中学习筛选信息会增加训练负担。
   - **对 wiki 的映射：** 感知输入裁剪、足点预测和训练效率。
2. **方法：** Foothold Prediction Model（FPM）只根据本体感觉和速度指令预测未来足点；策略只观察预测足点周围的小块高程图。FPM 与行走策略联合训练，并根据策略新生成的 rollout 周期性重训。
   - **对 wiki 的映射：** FPM 与 locomotion policy 的训练闭环，以及论文实体页中的流程总览。
3. **结果：** 论文报告达到相同地形难度约需完整高程图策略三分之一的训练迭代；在稀疏落脚点的狭窄围墙走廊和未见地形上优于完整地图策略，与训练成本更高的注意力编码器表现相当。
   - **对 wiki 的映射：** 效率和泛化评测，注意区分论文报告与跨任务结论。
4. **真机与本体：** 方法在 Unitree G1 上进行零样本迁移，实验包括障碍、楼梯、沟隙及下台阶，覆盖室内外及训练分布外情况。
   - **对 wiki 的映射：** G1 部署证据与感知式越障任务。

## 复现边界

论文给出项目页链接；训练与评估实现见官方 GitHub 仓库。当前代码 README 标注平台为 Unitree G1（29 DoF）与 Isaac Lab，并提供策略及 FPM 训练、播放和验证入口。项目页中的额外实验细节应以该页后续可访问内容为准。

## 关联来源

- [项目页归档](../sites/watch-your-step.md)
- [官方代码归档](../repos/foothold-prediction-model.md)
- [论文实体页](../../wiki/entities/paper-watch-your-step.md)
