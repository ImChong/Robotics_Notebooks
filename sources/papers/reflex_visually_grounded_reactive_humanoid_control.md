# Reflex: Visually Grounded Reactive Humanoid Control

> 来源归档（ingest）

- **标题：** Reflex: Visually Grounded Reactive Humanoid Control
- **类型：** paper / humanoid / reactive-catching / whole-body-control / sim2real
- **论文：** <https://arxiv.org/abs/2610.12772>（arXiv 预印本）
- **项目页：** <https://toberj.github.io/reflex/>
- **代码：** <https://github.com/ToberJ/reflex>（仿真训练 / 评测与 checkpoint 已公开，真机栈未包含）
- **作者：** Taoyang Jia、Weikai Huang、Linxin Song、Jared Darlington、Jieyu Zhao、Yue Wang、Jiafei Duan、Zhongzheng Ren、Ranjay Krishna
- **机构：** University of Washington、Stanford University、University of Southern California、National University of Singapore、University of North Carolina at Chapel Hill、Allen Institute for AI
- **平台：** Unitree G1，29 DoF；躯干 RGB-D 相机
- **入库日期：** 2026-10-12
- **一句话说明：** 以三阶段训练让 G1 从特权状态 PPO 接球策略，过渡到应对观测时延的动力学潜变量，再用 egocentric RGB-D 历史和机身运动估计潜变量，实现真实箱体接球。

## 摘录 1：任务与核心主张（项目页 / 论文）

- **任务：** 人类向人形机器人抛出箱体，机器人根据机载视觉和本体状态用全身动作接住；这是高速动态物体交互，对预测物体运动和补偿传感器时延有要求。
- **主张：** 采用「先学反应控制，再学在不完整状态下控制，最后接入视觉」的 curriculum。第三阶段冻结控制策略，仅训练视觉 encoder 复原策略需要的 dynamics latent。
- **平台与频率：** Unitree G1（29 DoF）；躯干相机输出 192×144 RGB-D；控制频率 50 Hz，Isaac Lab 仿真物理步进 200 Hz；策略输出关节位置目标。

**对 wiki 的映射：** 统一编译到 [Reflex 论文实体页](../../wiki/entities/paper-reflex-visually-grounded-reactive-humanoid-control.md)，并与 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)互链。

## 摘录 2：三阶段训练流程

1. **状态特权 catching policy：** 在仿真中用物体状态训练全身接球策略，PPO 学习把手与躯干移向拦截位置并稳定承接。
2. **动力学潜变量与时延鲁棒性：** 策略由不完整、延迟的物体观测推断其运动；在第一阶段策略基础上训练 dynamics latent，并通过 action matching / DAgger 减少状态策略与部署策略的差距。项目页报告观测时延消融的视觉历史收益。
3. **视觉 grounding：** CNN/GRU 编码 egocentric RGB-D 帧历史及相机自运动，回归特权策略所需的 dynamics latent；此阶段固定控制器，只训练视觉表征到 latent 的映射。

**对 wiki 的映射：** 流程图强调训练监督路径；公开仓实现了仿真视觉评测；真机部署栈未随仓库公开。

## 摘录 3：评测与证据边界

- **仿真：** RGB-D 视觉策略成功率 85.7%；特权观测对照为 87.9%。没有视觉历史时，居中 / 偏置抛掷成功率分别下降 35.6 / 39.8 个百分点。
- **真机 G1：** 2.5 m 与 5 m 抛掷距离成功率分别为 65% 与 40%；人类接球率分别为 85% 与 68.3%，机器人达到对应人类成功率的约 76% 与 59%。
- **距离范围：** 论文报告 7 m 真机成功率为 15%，对应人类成功率 28.3%；各距离真机样本量为 20 次。
- **证据边界：** 仿真最优策略接近特权观测上限；现实成功率随距离增加而下降。不同距离上的人类结果是参照，不是机器人任务成功的替代指标。

**对 wiki 的映射：** 分开报告模拟/真机、成功率与演示范围，指出观测历史消融显示视觉闭环的重要性，但真机较远距离仍有明显性能落差。

## 官方资料之间的身份关系

- 项目页提供论文 PDF、演示与代码仓入口。
- GitHub README 提供三阶段仿真训练、评测和 checkpoint；真机部署栈不在发布范围内。
- 论文为 arXiv:2610.12772；项目身份沿用原有稳定 `project_id`。
