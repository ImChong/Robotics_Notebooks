# Push the Limit: Learning High-Speed Sprinting and Aggressive Cornering on a Full-Scale Humanoid（arXiv:2610.14840）

> 来源归档（ingest）

- **标题：** Push the Limit: Learning High-Speed Sprinting and Aggressive Cornering on a Full-Scale Humanoid
- **类型：** paper / humanoid locomotion / high-speed sprinting / aggressive cornering / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2610.14840>
- **arXiv HTML：** <https://arxiv.org/html/2610.14840v1>
- **作者：** Huaxing Huang、Ruidong Li、Jinchao Han、Ziyang Dong、Wenhao Cui
- **机构：** 松延动力（Noetix Robotics）、清华大学（Tsinghua University）
- **发表：** CoRL 2026（论文页面标注已接收）
- **视频：** <https://youtu.be/YWf62XkiApc>
- **入库日期：** 2026-10-25
- **一句话说明：** 针对全尺寸双足人形高速冲刺时本体感觉历史受步频振荡污染、偏航角速度奖励妨碍转弯、多技能共享 AMP 判别器导致技能塌缩的问题，提出因果多头注意力历史编码、航向方向奖励和技能条件 Multi-AMP；论文报告仿真 6.5 m/s 指令跟踪与真机 100 米平均 7.14 m/s。

## 摘要级要点

- **平台与任务：** 1.8 m、17-DoF 双足人形；统一策略覆盖站立、原地转向、行走和跑步，策略以 50 Hz 部署，输出关节角目标并由 PD 执行。
- **高频历史编码：** 因果多头注意力从本体感觉历史中提取时序动态，在相近参数规模下替代把历史直接展平输入 MLP；消融显示结构化时序编码支持高速跟踪，MHA 在高速激进转向中表现更稳。
- **转弯奖励：** 高速步态的躯干摆动会污染瞬时偏航角速度。方法在高速区改用基座速度方向相对目标航向的误差监督低频转向趋势，低速区仍使用偏航角速度跟踪。
- **技能条件 Multi-AMP：** 为站立、原地转向、行走、跑步分别使用动作风格判别器，并按命令技能选择风格奖励；共享判别器的消融使原地转向退化为近似静止。
- **评测与边界：** Gazebo 仿真跟踪 6.5 m/s 速度指令并测试转弯；真机标准跑道 100 m 计时实验在 7.0 m/s 指令下用时 14 s，平均 7.14 m/s。该实验采用减重版主平台（保留相同执行器与运动学）；论文指出高速关节接近力矩上限，抗扰余量有限。
- **开放状态：** arXiv HTML 列有演示视频，但页面未列代码仓库或数据下载链接；目前按未见官方可运行实现归档。

## 对 wiki 的映射

- 沉淀实体页：[Push the Limit：全尺寸人形高速冲刺与激进转弯](../../wiki/entities/paper-push-the-limit-humanoid-sprinting.md)
- 交叉更新：[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[AMP & HumanX](../../wiki/methods/amp-reward.md)、[人形机器人运控策略的观测输入](../../wiki/concepts/humanoid-policy-observation-inputs.md)
