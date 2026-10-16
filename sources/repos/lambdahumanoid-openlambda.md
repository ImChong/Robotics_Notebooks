# LambdaHumanoid/openlambda

- **URL：** <https://github.com/LambdaHumanoid/openlambda>
- **项目页：** <https://lambdahumanoid.github.io/>
- **主论文：** [λ₀（arXiv:2610.18950）](../papers/lambda0_arxiv_2610_18950.md)
- **关联论文：** [早期技术报告（arXiv:2610.00438）](../papers/lambda0_arxiv_2610_00438.md)
- **模型权重：** <https://huggingface.co/lambda-humanoid/lambda0-midtrain-100k>
- **许可证：** Apache-2.0（仓库 README 徽章）
- **开源状态：** 部分开源；代码、Stage II 100k checkpoint 和少量动作对齐示例可用，HumanVerse-500 数据与完整 Stage III post-training / 实机部署流程未发布。

## 一句话说明

OpenLambda 为人形 loco-manipulation 提供人类数据采集、同步、动作对齐、仿真回放及 λ₀ 模型训练和推理代码。

## 仓库公开范围

- **采集与处理：** 人类穿戴设备采集、视频/动作同步、动作对齐、仿真回放。
- **模型：** λ₀ 的 Stage I/II/III 训练代码、配置示例和 inference server；README 提醒 Stage I/II 不保证端到端复现。
- **权重：** Stage II mid-training checkpoint 已发布。
- **数据：** HumanVerse-500 与预训练/mid-training 数据不包含在仓库；目前只提供少量动作对齐示例。
- **缺口：** G1 episode recorder、原始人类/机器人数据转换，以及完整 Stage III、仿真评测和真机部署流程仍标为待完善；推理服务只返回动作预测，不直接连接机器人控制桥。

## 对 wiki 的映射

- [λ₀ / HumanVerse-500 实体页](../../wiki/entities/paper-lambda0-egocentric-human-pretraining.md)
