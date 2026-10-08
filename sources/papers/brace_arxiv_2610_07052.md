# BRACE（arXiv:2610.07052）

> 来源归档（paper；2026-10-08）

- **标题：** BRACE: Adapting Whole-Body References for Force and Terrain Aware Humanoid Motion Tracking
- **论文：** <https://arxiv.org/abs/2610.07052>
- **HTML：** <https://arxiv.org/html/2610.07052v1>
- **项目页：** <https://multiplylabs.github.io/brace/>
- **作者：** Sudarshan Harithas、Chen Yu、Juan Borbon、Shubhankar Mondal、Winston Zha、Srinath Sridhar、Dingqi Zhang、Jiuguang Wang
- **机构线索：** 论文将 Multiply Labs 项目页列作 affiliation link；致谢提到 Brown University 与 University of New Mexico。
- **代码状态：** 发现的 GitHub 仓库是项目页源码；README 说算法代码链接待就绪。
- **对 wiki 的映射：** [BRACE](../../wiki/entities/paper-brace-force-terrain-whole-body-tracking.md)

## 核心摘录

1. **问题：** 平地、空载的人体参考没有写出机器人在坡面、负载和手部接触下应当交换的 wrench。
2. **地形参考变换：** 用高度图调足端局部高度与根部，配合足部朝向和 IK 得到 terrain-conformed reference。
3. **wrench 参考变换：** 施力模式考虑手臂 effort 限制、手部 lead 与 CoM/CoP brace；补偿模式跟踪地形参考并抵抗外部 wrench。
4. **双教师蒸馏：** Teacher-E/Teacher-C 经 DAgger 汇入 flow-matching student；student 用 proprioception、reference、mode 和 wrench command 推断，不依赖部署高度图或测得 wrench。
5. **验证：** Unitree G1 仿真和真机实验覆盖施力、补偿、坡面及应用任务。
6. **开放状态：** 项目页 repo 的 README 标明 code link 待就绪；当前没有可验证的算法源码/训练入口。

