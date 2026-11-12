# Runway Praxis-1 官方技术报告与发布页
> 来源归档（project / site） · 2026-11-12

- **项目/技术报告：** https://runway.com/research/praxis-1-video-pretraining-to-robot-control
- **发布说明：** https://runway.com/news/company-news/praxis-1-is-now-available
- **代码：** https://github.com/runwayml/praxis-1-oss
- **模型权重：** https://huggingface.co/collections/runwayml/praxis-1
- **HF LeRobot 文档：** https://huggingface.co/docs/lerobot/main/en/praxis
- **许可：** 公开仓库与官方发布页声明 Apache-2.0；使用以实际各文件/模型许可证为准。
- **核对范围：** 技术报告 §1–§13、官方发布与资源入口。
- **通读：** 已完成

## 开放状态与边界

- **已开源：** 官方源码库包含模型定义、推理、WebSocket policy server、MuJoCo 仿真及全参数/LoRA 微调；HF 提供 `praxis-1-base`、`praxis-1-droid`、`praxis-1-yam` 权重与 LeRobot 适配。
- **复现边界：** 默认 bf16 模型约 54.4 GB，官方快速开始要求单张至少 80GB CUDA GPU；不同本体需匹配相机、位姿、归一化及动作头。Base 权重无本体动作投影，不能直接当部署策略。
- **量化边界：** 官方仓库 NVFP4 routed expert 选项仅 SM120，README 明确量化闭环质量不一定与 bf16 一致；权重缩小不等于所有 GPU 均可执行。

## 对 wiki 的映射

- [Praxis-1](../../wiki/entities/runway-praxis-1.md)
- [技术报告原文归档](../papers/runway_praxis_1_technical_report_2026_11.md)
- [源码归档](../repos/runway-praxis-1-oss.md)
