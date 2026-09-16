# TextOp（TeleHuman）

- **标题：** TextOp: Real-time Interactive Text-Driven Humanoid Robot Motion Generation and Control
- **论文：** <https://arxiv.org/abs/2602.07439>
- **项目页：** <https://text-op.github.io/>
- **代码：** <https://github.com/TeleHuman/TextOp>
- **许可证：** MIT
- **类型：** paper / text-driven-humanoid-control
- **机构：** 中国电信人工智能（TeleAI）、上海交通大学（SJTU）、华东理工大学（ECUST）
- **收录日期：** 2026-09-16
- **开源结论：** **已开源** — 训练、推理与 G1 部署脚本齐全；README 提供预训练 RobotMDAR 与 Tracker 权重及公开数据集处理流程

## 一句话摘要

端到端开源管线：高层 `TextOpRobotMDAR`（文本→自回归运动扩散）+ 低层 `TextOpTracker`（BeyondMimic 系通用跟踪）+ `TextOpDeploy`（MuJoCo sim2sim 与 G1 sim2real）；支持流式改令与模块化扩展。

## 仓库结构（README）

```
TextOp/
├── TextOpRobotMDAR/   # 高层 text-to-motion（VAE + LDM）
├── TextOpTracker/     # 低层全身通用跟踪策略
├── TextOpDeploy/      # sim2sim / sim2real 部署
├── dataset/           # 数据集处理脚本
├── deps/              # 第三方依赖
└── docs/
```

## 复现入口

- 使用说明：[USAGE.md](https://github.com/TeleHuman/TextOp/blob/main/USAGE.md)
- 依赖公开 AMASS+BABEL（及 LAFAN1 等）与 GMR 重定向；官方称仅用公开数据亦可达到相近性能
- 跟踪基于 [BeyondMimic](https://beyondmimic.github.io/)；高层扩散架构改编自 [DART](https://github.com/zkf1997/DART)

## 交叉链接

- [TextOp 论文归档](../papers/textop_arxiv_2602_07439.md)
- [TextOp 项目页](../sites/textop.md)
- [TextOp 论文实体](../../wiki/entities/paper-loco-manip-161-022-textop.md)
