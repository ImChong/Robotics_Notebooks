# CoAWM 与 Gym-Anything：待完成的资料收集

> 状态：草稿。尚未通读两篇论文全文，也未阅读 Gym-Anything 可运行源码；不得视为完成 ingest。

## CoAWM

- 论文：[Continuous-Action World Models: Learning Cross-Embodiment Spatiotemporal Dynamics Priors](https://arxiv.org/abs/2610.40401)，arXiv:2610.40401v1，2026-10-29。
- 作者：Liangyu Li、Qingwen Liu。
- 摘要提出以共享潜在函数空间表示不同机器人和采样频率的动作；预训练后由观测、语言和状态条件化预测动作函数，并按查询时间解码末端动作。
- 待办：通读全文与必要附录，核实实验、局限及官方代码状态；按仓库规范编写实体页和项目流程图。

## Gym-Anything / RoboWorld

- 论文：[Gym-Anything for Robotics: Scaling physical simulations with digital agents](https://arxiv.org/abs/2610.40565)，arXiv:2610.40565v1，2026-10-29。
- 作者：Pranjal Aggarwal、Sean Welleck；卡内基梅隆大学。
- [项目页](https://gym-anything.github.io/robotics/)；[官方源码](https://github.com/gym-anything/robotics)，仓库声明 MIT 许可。
- 项目页称通过创建和审计智能体构造仿真及任务；RoboWorld 包含 25 个仿真、25 个测试和 175 个训练环境。
- 待办：通读论文全文与必要附录，阅读官方运行入口及相关源码，分别归档项目页和代码，编写实体页、项目流程图与源码运行时序图，核查实验及局限。

## 验证状态

尚未运行 `make ci-preflight`；当前环境无法直接克隆仓库。本文件仅保存已核实线索和未完成事项。
