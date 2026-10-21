# OpenLambda（λ₀）官方源码核查

> 来源归档（repo / 官方实现）

- **仓库：** <https://github.com/LambdaHumanoid/openlambda>
- **组织：** LambdaHumanoid
- **默认分支：** `main`
- **可见性：** public
- **许可证：** Apache-2.0（仓库声明）
- **项目页：** <https://lambdahumanoid.github.io/>
- **论文：** <https://arxiv.org/abs/2610.18950>
- **模型：** <https://huggingface.co/lambda-humanoid/lambda0-midtrain-100k>
- **项目页归档：** [lambda-humanoid-project-page.md](../sites/lambda-humanoid-project-page.md)
- **Wiki 主实体：** [λ₀ / OpenLambda](../../wiki/entities/paper-lambda0-egocentric-human-pretraining.md)
- **核查日期：** 2026-10-16

## 仓库覆盖范围

README 将 OpenLambda 定位为从人类经验到人形基础模型的开放工具集，目录覆盖硬件与环境配置、人类数据采集、动作/视觉对齐、仿真回放、λ₀ 模型及训练包。机器人为 Unitree G1（29 个身体自由度，不含手），执行端为 SONIC 全身控制器和 BrainCo Revo2 手。

| 环节 | README / 仓库入口 | 当前状态 |
|------|------------------|----------|
| 人类采集 | `data_collection/human`、`setup/human` | 有设备与采集流程说明 |
| 动作对齐 | `data_alignment/action_alignment` | 仓库含对齐模块；按当前 README 使用提供的数据转换流程 |
| 模型训练/推理 | `lambda0` | 代码及示例配置部分公开；权重发布为 Stage II 100k checkpoint |
| Stage III 机器人后训练 | README 的 Workflow & Release Checklist | 完整流程明确标为准备中 |
| HumanVerse 数据 | README、HF 页面 | 全量数据集未包含在仓库；样例数据链接当前返回 401 |

## 开源与复现边界

- 代码和 checkpoint 按项目声明使用 Apache-2.0；完整的人类预训练/中训练数据不随仓库发布。
- `lambda0-midtrain-100k` 含模型配置与 2B 参数权重，是 Stage II 结束时的中训练 checkpoint，不含 EMA 权重、任务特定 Stage III 策略，不能直接视为真机可运行模型。
- 仓库近期仍在发布推进中。复现者应逐项核对 workflow checklist，尤其是 Stage III 的数据、训练与部署脚本是否已补齐。
