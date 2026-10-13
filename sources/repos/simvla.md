# SimVLA 官方代码仓库

> 来源归档

- **类型：** repo
- **仓库：** <https://github.com/kyounginbaik/SimVLA>
- **许可证：** BSD-3-Clause
- **论文：** <https://arxiv.org/abs/2610.11248>
- **项目页：** <https://kyounginbaik.github.io/simvla/>
- **入库日期：** 2026-10-13
- **一句话说明：** 官方仿真与数据工作流入口，覆盖场景、任务/技能、SimAction/SimVQA 数据与部分评测；不是含训练配方和权重的端到端复现仓库。
- **沉淀到 wiki：** [SimVLA 论文实体](../../wiki/entities/paper-simvla-zero-shot-mobile-manipulation.md)

## 仓库可见能力

README 与文档列出 Python 3.10–3.12 安装入口、CPU 程序化场景/任务检查、仿真数据生成、回放与评估说明。完整仿真依赖 Isaac Sim 4.5、Isaac Lab、cuRobo 和额外机器人/场景资产；CPU quickstart 不代表完整 rollout 可在 CPU 上复现。项目页还链接数据与资产资源。

## 复现边界

- **未包含：** 完整 SimVLA 训练 recipe、SimDeploy policy-training adapter、论文 checkpoint。
- **可复用：** 场景与任务设施、轨迹/VQA 数据工作流、部分回放和评测工具。
- **数据：** 项目页列有 SimVLA_data 相关集合，包括 Anubis mug-to-sink LeRobot 数据与 BODex/资产资源；内容和版本以 Hugging Face 上游为准。
- **判断：** 对仿真建任务与数据有用，但仅凭当前仓库无法完整复刻论文学习结果。

## 与知识库的关系

| 主题 | 关系 |
|---|---|
| [VLA](../../wiki/methods/vla.md) | 仿真预训练与 rollout 后训练实例 |
| [Manipulation](../../wiki/tasks/manipulation.md) | 厨房家务与长程移动操作 |
| [论文实体](../../wiki/entities/paper-simvla-zero-shot-mobile-manipulation.md) | 方法、指标与局限 |
