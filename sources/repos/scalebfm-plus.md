# ScaleBFM_Plus

- **URL：** <https://github.com/Hxxxz0/ScaleBFM_Plus>
- **类型：** repository fork / ScaleBFM implementation extension
- **入库日期：** 2026-10-25
- **上游仓库：** <https://github.com/zengweishuai/ScaleBFM>
- **主论文：** [Scaling Behavior Foundation Model for Humanoid Robots](https://arxiv.org/abs/2607.15163)（arXiv:2607.15163）
- **项目页：** <https://scalebfm.github.io/>（Code 按钮指向上游仓库）
- **开源状态：** 可访问的 fork，包含 ScaleRetarget、ScaleTrack 与 ScaleBridge；仓库 API 未声明许可证。该 fork 不是项目页链接的上游官方仓库。
- **Tags：** #bfm #behavior-foundation-model #humanoid #motion-tracking #supermimic #unitree-g1 #onnx

## 一句话说明

ScaleBFM_Plus 是 ScaleBFM 上游代码的 fork；除 retarget、PPO 训练和部署流程外，维护者新增了基于可微世界模型的 SuperMimic-style 训练、公开 checkpoint 与 ONNX 导出路径。

## 仓库要点（2026-10-25 抓取）

| 字段 | 内容 |
|------|------|
| 上游关系 | GitHub 将该仓库标记为 `zengweishuai/ScaleBFM` 的 fork |
| 最近新增 | SuperMimic-style training、released M checkpoint、ONNX export 与对比结果 |
| 训练实现 | README 描述 learned differentiable world model、32-step imagined BPTT 与 return-based reset |
| 速度说明 | 仓库提交与 README 报告单步约 0.65 s，相对 PPO iteration 17.0 s；作者称约 26×。这是仓库自报数据，复现实验配置需以项目说明为准 |
| checkpoint | README / 提交记录提及 SuperMimic-M checkpoint 与 ONNX 文件 |
| 许可证 | GitHub 仓库元数据未声明 |

## 对 wiki 的映射

- [ScaleBFM 论文实体](../../wiki/entities/paper-scaling-bfm-humanoid.md)
- [BFM 概念页](../../wiki/concepts/behavior-foundation-model.md)

## 维护备注

- 官方项目页的 **Code** 链接仍指向上游 [zengweishuai/ScaleBFM](https://github.com/zengweishuai/ScaleBFM)；本条记录的是用户提供的 fork，不将其新增内容归为论文作者的官方结果。
- 上游 README 说明 ScaleTrack 的训练、ScaleRetarget 的动作重定向和 ScaleBridge 的 Sim2Sim / Sim2Real 部署入口；fork 的近期提交补入 SuperMimic-style 训练与模型导出。
- SuperMimic checkpoint 的“与官方 ScaleBFM-M 持平”及训练速度数据均为仓库维护者报告，不等同于论文原文结论。
