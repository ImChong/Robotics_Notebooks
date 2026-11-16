# Alloy: A Multi-Material Physics Engine for Robot Learning｜Alloy：多材料机器人物理引擎

- **类型：** 研究原始论文
- **论文：** https://arxiv.org/abs/2611.16312 （v1）
- **HTML：** https://arxiv.org/html/2611.16312v1
- **PDF：** https://arxiv.org/pdf/2611.16312
- **项目：** https://alloy.argonrobotics.ai/docs/
- **代码：** https://github.com/argonrobotics-ai/alloy
- **核对范围：** arXiv 原始版本的研究任务、核心方法与算法、主要实验评测、结论及限制；不将第三方摘要作为通读依据。没有在本次执行模型训练或真机复现。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文详情](../../wiki/entities/paper-alloy-multimaterial-2611-16312.md)

## 方法与评测核对

- **研究问题：** 依据材料卡将刚体、软体、颗粒和液体交给不同求解器，并在单一场景中耦合仿真。
- **方法原理：** Scene 统一读取材料与几何对象，Router 按物相/形态/刚度选 MuJoCo、Newton VBD/MPM 或 SPH，Coupler 在控制步内部交换接触影响，并用每世界健康分级检查数值稳定性。
- **原文结果：** 官方文档支持一份场景描述混合 MuJoCo 机器人、水和柔性材料；并行数与速度依赖 GPU、材料和耦合求解设置。
- **局限与范围：** 各类材料的参数、求解步长和物理精度边界不同；健康度 OK 只说明求解器运行稳定，不能代替实物物性校准。
