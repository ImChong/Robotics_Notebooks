# RISE：面向感知式人形机器人起身的测试时运动引导（arXiv:2610.30061）

> 来源归档（ingest）

- **标题：** Test-Time Motion Steering for Perceptive Humanoid Fall Recovery
- **简称：** RISE
- **类型：** paper / humanoid / fall-recovery / perception / motion-steering
- **arXiv：** <https://arxiv.org/abs/2610.30061>（v1，2026-10-21）
- **项目页：** <https://osheraz.github.io/RISE/>
- **作者：** Osher Azulay、Cheng-Lin Hsieh、Stella X. Yu（前两位共同一作）
- **平台：** Unitree G1（项目页与论文摘要）
- **代码：** 项目页标注 “Code Coming soon”；截至 2026-10-23 未提供可用代码链接
- **入库日期：** 2026-10-23
- **一句话说明：** 通过预测未来身体状态并在推理时用任务代价引导运动意图，让 G1 能按目标朝向、起身速度或路径选择起身方式。

## 摘要级要点

- **问题：** 同一倒地姿态可以通过多种动作起身；可靠恢复之外，还需根据障碍、地形和任务选择合适的恢复路径。
- **方法：** 从可复用的恢复动作构建动作词汇；视觉扩散规划器预测未来身体状态，任务代价在测试时引导方向、速度或路径，再据此生成运动 latent 并通过反馈执行。
- **感知与适应：** 项目页展示第一人称深度观测条件下的地形适应，以及靠墙时朝无遮挡方向引导的恢复。
- **实验：** 论文摘要报告仿真中的多样化恢复、地形适应和无需重训的测试时引导，并展示 Unitree G1 真机结果。
- **开源状态：** 官方项目页目前写明代码即将发布，但未给出仓库 URL；勿据此当作已开源实现。

## 核心论文摘录（MVP）

### 1) 预测状态，再生成可执行动作

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.30061)；项目页 “How it works”
- **摘录要点：** RISE 将执行分为观察、未来身体状态、运动 latent、马达动作；以状态意图作为任务引导接口，再由反馈解码器实现动作。

### 2) 测试时任务引导

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.30061)；项目页 “Test-time steering”
- **摘录要点：** 任务代价可引导预期朝向、起身速度或恢复路径，不需为每个目标重新训练策略。

### 3) 开源状态核查

- **链接：** [RISE 项目页](https://osheraz.github.io/RISE/)
- **摘录要点：** 项目页列出论文与视频，代码状态标注为 “Code Coming soon”；截至入库日未列可访问代码仓库。
- **对 wiki 的映射：** 待实体页升格后链接本归档及 [RISE 项目页归档](../sites/osheraz-rise.md)。

## 对 wiki 的映射

- 计划沉淀实体页：`wiki/entities/paper-rise-humanoid-fall-recovery.md`（待建）
- 项目页归档：[`sources/sites/osheraz-rise.md`](../sites/osheraz-rise.md)
- 相关主题：人形机器人跌倒恢复、感知式全身控制、测试时规划
