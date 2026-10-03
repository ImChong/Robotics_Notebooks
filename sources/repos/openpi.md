# openpi（Physical Intelligence）

> 来源归档（官方代码仓）

- **标题：** openpi — Open-source models and packages for robotics
- **类型：** repo
- **链接：** https://github.com/Physical-Intelligence/openpi
- **关联论文：** [π0.5](https://arxiv.org/abs/2504.16054) · [Knowledge Insulation](https://arxiv.org/abs/2505.23705)
- **关联项目页：** <https://www.pi.website/blog/pi05>
- **入库日期：** 2026-10-03
- **一句话说明：** Physical Intelligence 官方机器人模型仓库，包含 π₀、π₀-FAST 与 π₀.₅ 的代码、训练/推理示例及公开 checkpoint。

## 开源核查（截至 2026-10-03）

- **代码：** 已公开。README 将 π₀.₅ 列为仓库支持模型；支持 flow-matching head 的训练与推理，并提供 LIBERO 微调配置和机器人推理服务入口。
- **权重：** 已公开。README 列出 π₀.₅ base checkpoint（gs://openpi-assets/checkpoints/pi05_base）及 π₀.₅-LIBERO、π₀.₅-DROID 微调 checkpoint。
- **数据与完整训练配方：** 未全部公开。README 对模型预训练规模作出说明，但本仓库没有发布组成预训练数据的完整数据集。
- **Knowledge Insulation 边界：** README 说明 π₀.₅ 权重经过 Knowledge Insulation 训练；当前公开微调代码支持 flow-matching head，但没有实现论文所述用 FAST 目标更新 VLM 骨干的 KI 微调方案。细节见 [论文来源归档](../papers/knowledge_insulation_arxiv_2505_23705.md) 及 [官方 issue #649](https://github.com/Physical-Intelligence/openpi/issues/649)。

## 主要入口

| 用途 | 仓库入口 |
|---|---|
| 通用模型与基础 checkpoint | README 的 Model Checkpoints |
| π₀.₅-LIBERO 微调 | pi05_libero 配置、scripts/train.py |
| 推理与动作块生成 | policy.infer(example)["actions"] |
| 独立策略服务 | scripts/serve_policy.py；仓库另有远程推理示例 |
| PyTorch 版本 | README 的 PyTorch Support（π₀ 与 π₀.₅，限制见官方文档） |

> checkpoint 可作为微调/推理起点；接入新机器人仍需适配观测、动作空间、归一化和执行接口。仓库示例不等于任意机器人开箱即用。

## 延伸与策展

- [Humanoid Motion Intelligence 开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 将 openpi 归入「世界模型、VLA与Agent」路线。
- [openpi-rtc.md](openpi-rtc.md) 记录社区 Real-Time Chunking 实现；官方 openpi 栈未内置 RTC 一等入口，LeRobot 文档见 [lerobot-rtc-docs.md](../sites/lerobot-rtc-docs.md)。

## 对 wiki 的映射

- [π0.5 论文实体](../../wiki/entities/paper-pi05-open-world-vla.md)
- [π₀ 论文实体](../../wiki/entities/paper-pi0.md)
- [π₀ 策略方法页](../../wiki/methods/π0-policy.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
- [χ₀ / kai0](../../wiki/entities/paper-kai0.md) — 基于 openpi 的协同叠衣后训练与部署对齐
- [Jetson Thor π₀.₅ 教程](../courses/jetson_openpi_pi05_on_thor.md)
- [Knowledge Insulation 论文实体](../../wiki/entities/paper-knowledge-insulation.md)
