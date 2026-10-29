# NavGPT-3 官方代码

- **代码：** <https://github.com/metacognitionai/NavGPT-3>
- **项目页：** <https://metacognitionai.github.io/NavGPT3/>
- **论文：** <https://arxiv.org/abs/2610.10787>
- **许可：** Apache-2.0；组件、模型及数据另有许可，参见仓库 NOTICE。
- **核查日期：** 2026-10-29

官方 README 标注已发布 VLA 推理服务、harness、R2R-CE / RxR-CE 仿真评测、配置与 4B / 8B 权重；基于 ZEOS 的 Unitree 部署代码待发布。复现需要 Matterport3D 场景授权、数据、GPU 和对应环境；完整 planner 还需对应模型账号或 API。

运行入口：设置 `configs/paths.yaml` 后，`navgpt launch --config CFG` 启动 Habitat 与 VLA 服务并执行评测；也可分别运行 `python -m navgpt.environment`、`python -m navgpt.vla` 和 `navgpt run`。

关联：[论文归档](../papers/navgpt3_arxiv_2610_10787.md) · [项目页归档](../sites/navgpt-3.md) · [实体页](../../wiki/entities/paper-navgpt-3.md)。
