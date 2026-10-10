# SGS 官方项目页：Success-Guided Sampling

- **项目页：** https://sgs-rl.github.io/
- **论文：** [A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control](https://arxiv.org/abs/2610.12465)（arXiv:2610.12465）
- **作者：** Octi Zhang、Mateo Guaman Castro、Patrick Yin、Ignacio Dagnino、Abhishek Gupta、Rosario Scalise、Byron Boots
- **机构：** University of Washington；NVIDIA（作者脚注）
- **代码：** 官网当前为 “Code (coming soon)”，暂无可验证的 GitHub 仓库链接；截至 2026-10-10 未开源可运行实现。
- **入库日期：** 2026-10-10

## 项目页内容摘录

SGS（Success-Guided Sampling）通过追踪策略在任务配置上的近期成功率，以自适应权重选择训练 episode 的 reset 配置。目标不是均匀消耗 rollout 在已掌握或当前无法企及的设置，而是聚焦策略能力前沿。核心策略优化仍为标准 PPO，采样器位于 PPO 外层。

项目页将 task configuration 定义为初始状态 $s_0$、目标 $g$、环境 $e$ 的三元组。locomotion 覆盖 ANYmal C/D 与复杂地形；manipulation 覆盖 UR5e / Franka 的精细装配。页面展示仿真训练后转为 RGB policy，并在 UR5e 真机执行 assembly 的 zero-shot transfer。所有演示视频页面标注为 1× 速度。

## 公开结果摘录

- ANYmal D 多地形：从 4K 到 1M 并行环境，SGS 成功率由 0.46 提至 0.73；对照方法在最高规模表现明显下降或停滞。
- Franka nut-and-bolt：1M 并行环境时 SGS 为 0.70 ± 0.19；PLR 为 0.05 ± 0.11，Uniform 为 0.06 ± 0.13。
- 上述为官网公开表格结果，完整实验定义、置信区间与消融应回到论文正文核对。

## 开放状态边界

官网论文链接已指向 arXiv:2610.12465，会议标注 CoRL 2026；代码明确显示 “coming soon”。因此当前可引用官方演示与论文，但不可声称训练/部署代码已开放，也不应将“无真实数据 / 无 demonstrations”简化成“训练完全没有数据”：策略仍从模拟器采样生成交互经验，承诺重点是无需人类演示或逐任务 reward engineering。