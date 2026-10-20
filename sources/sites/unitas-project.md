# UNITAS 项目页（unitas-wam.github.io）

> 来源归档

- **标题：** UNITAS: 3D-Native World Action Model
- **类型：** site（项目页）
- **URL：** <https://unitas-wam.github.io>
- **论文：** <https://arxiv.org/abs/2610.12099>
- **代码：** <https://github.com/DexForce/UNITAS>
- **入库日期：** 2026-10-12（项目页初次收录）；**本次核验：** 2026-10-20
- **一句话说明：** 官方项目页展示策略模式、世界模型模式、架构、基准结果，并提供可旋转视角、拖动时间轴的 3D 场景预测浏览；作者单位为香港中文大学（深圳）、DexForce、佛山大学、中山大学、华南理工大学。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Policy 模式 | 给定语言指令生成任务条件夹爪运动与可执行指令；只要动作时可跳过 flow 生成 |
| World-model 模式 | 给定夹爪轨迹，预测被查询场景点在同一时间区间内的运动 |
| 架构 | 世界对齐 3D 位置编码、单 token 轨迹 tokenizer、场景流与机器人动作的耦合专家 |
| 结果 | LIBERO 99.8%、LIBERO-Plus 89.1%、RoboTwin Randomized 93.04%、VLABench 56.2%；1.7B 参数 |

## 官方资源与开源核查（2026-10-20）

- 官方项目页当前提供交互式 3D 预测浏览：可旋转视角并沿时间轴查看 action flow / scene flow；这不是把生成视频转换为交互式 3D 场景。
- 官方 GitHub README 已有安装步骤、场景流推理与 Viser 可视化示例，以及 LIBERO、LIBERO-Plus、RoboTwin、VLABench 评测指南；仓库含 `src/unitas`、`evaluation`、`docs` 等目录，许可为 Apache-2.0。
- 官方 README 将模型检查点与推理/评测代码标为已发布；训练代码和训练数据集仍标注未发布。模型权重托管于 Hugging Face：<https://huggingface.co/RobbinWang123/Unitas>。
- README 同时链接作者维护的项目报告页：<https://dexforce.com/technical-report/#/unitas-wam>。
