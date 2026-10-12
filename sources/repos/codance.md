# CoDance（generalroboticslab/codance）

> 官方源码仓库来源归档

- **标题：** CoDance: Learning Reactive and Compliant Human-Humanoid Interaction from Video
- **类型：** repo
- **链接：** <https://github.com/generalroboticslab/codance>
- **项目页：** <https://generalroboticslab.com/CoDance> — [归档](../sites/codance-generalroboticslab.md)
- **论文：** <https://arxiv.org/abs/2610.05324> — [归档](../papers/codance_arxiv_2610_05324.md)
- **数据：** <https://huggingface.co/datasets/GeneralRoboticsLab/CoDance>
- **许可证：** 根项目 Apache-2.0；vendored third_party/rsl_rl 为 BSD-3-Clause；从 HUSKY 改编的 AMP 模块标注 CC BY-NC 4.0
- **平台：** Python；Linux、NVIDIA GPU；uv 管理环境
- **入库日期：** 2026-10-11
- **一句话说明：** CoDance 官方可运行实现，包含多连杆顺应数据增强、mjlab/rsl_rl 训练与评测、发布策略及 Unitree G1 真机部署代码。
- **沉淀到 wiki：** [CoDance 实体页](../../wiki/entities/paper-codance.md)

## 仓库入口（README 摘要）

| 目标 | 官方入口 |
|---|---|
| 环境安装 | Linux + NVIDIA GPU + uv；README 给出 make sync 与 HF 数据下载命令。 |
| 下载数据与权重 | Hugging Face 数据集；README 称约 850 MB，含训练 clip pools、七个策略检查点和 ONNX 导出。 |
| 回放策略 | mjlab.scripts.run play --ckpt <checkpoint>。 |
| 单策略训练 | mjlab.scripts.run train --config src/mjlab/tasks/codancing/config/g1/conf/decoupled.yaml。 |
| 评测 | mjlab.scripts.run eval 与 scripts/clip_metrics.py。 |
| 数据生成 | datagen.justfile 中 multilink-rc-paper 和 multilink-manifest 配方。 |
| 真机部署 | src/mjlab/tasks/codancing/simple_deploy/README.md；ONNX runner、Vicon 与 Unitree SDK。 |

## 实现路径

仓库是 mjlab / MuJoCo Warp 项目分支，带 vendored rsl_rl 的 adversarial motion prior 训练扩展。训练配置位于 src/mjlab/tasks/codancing/config/g1/conf/；任务逻辑位于 src/mjlab/tasks/codancing/。解耦策略以 AMP 风格下半身目标配合上半身参考跟踪；wholebody 配置直接跟踪全身参考。仓库还提供多种基线与消融，包括无对抗项、无 clip chaining、stiff tracker 等。

实机路径由 simple_deploy.codance 运行器消费 ONNX 策略，不需在机器人电脑运行仿真环境；舞伴观测来自 live Vicon。部署文档包含网络接口、marker/profile 标定、遥控器检查、状态转换与停机机制。真实机器人部署须按其 operator procedure 操作，不能仅凭模拟器回放验证安全性。

## 可复现性与限制

- **代码：已开源。** README 给出配置、运行命令、评测与数据生成入口，可从发布检查点开始回放和评测。
- **数据/权重：可获取。** 仓库说明从 Hugging Face 下载；本归档未独立检查文件清单与数据许可细目。
- **资源：** 默认训练配置声明 16,384 并行环境；README 报告单卡训练且提供减小环境数的覆盖配置。完整训练比单次评测昂贵。
- **许可：** 不应把根目录 Apache-2.0 推广到整个仓库；vendor 与改编 AMP 文件带独立许可，尤其 CC BY-NC 4.0 有非商业限制。
- **研究边界：** 项目训练源头是单段双人舞视频，交互与舞伴分布较窄；部署依赖 Vicon 追踪。

## 交叉链接

- Wiki 实体：[paper-codance.md](../../wiki/entities/paper-codance.md)
- 项目页归档：[codance-generalroboticslab.md](../sites/codance-generalroboticslab.md)
- 论文归档：[codance_arxiv_2610_05324.md](../papers/codance_arxiv_2610_05324.md)
