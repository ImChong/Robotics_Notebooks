# Alloy: A Multi-Material Physics Engine for Robot Learning 官方源码核对

- **类型：** 官方源码仓
- **代码：** https://github.com/argonrobotics-ai/alloy
- **论文：** https://arxiv.org/abs/2611.16312
- **公开运行入口：** `python -c 'import alloy; sim=alloy.build(alloy.Scene(dt=0.01)); sim.run(1.0)'`
- **相关模块：** `alloy.Scene、alloy.build 与 Simulation`
- **运行产物：** state、health、render
- **核对范围：** 官方 README、该仓库公开入口脚本与关键模块的静态核查；外部数据和模型依赖未执行。
- **通读：** 已完成
- **wiki：** [论文独立详情](../../wiki/entities/paper-alloy-multimaterial-2611-16312.md)
- **原文：** [论文来源](../papers/alloy-multimaterial-2611-16312.md)

各类材料的参数、求解步长和物理精度边界不同；健康度 OK 只说明求解器运行稳定，不能代替实物物性校准。
