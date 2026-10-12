# UNITAS 项目页（unitas-wam.github.io）

> 来源归档

- **标题：** UNITAS: 3D-Native World Action Model
- **类型：** site（项目页）
- **URL：** <https://unitas-wam.github.io>
- **论文：** <https://arxiv.org/abs/2610.12099>
- **代码：** <https://github.com/DexForce/UNITAS>
- **入库日期：** 2026-10-12
- **一句话说明：** 官方项目页，展示策略模式、世界模型模式、架构与基准结果，附 3D 场景交互浏览；单位为香港中文大学（深圳）、DexForce、佛山大学、中山大学、华南理工大学。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Policy 模式 | 给定语言指令生成任务条件夹爪运动与可执行指令；只要动作时可跳过 flow 生成 |
| World-model 模式 | 给定夹爪轨迹，预测被查询场景点在同一时间区间内的运动 |
| 架构 | 世界对齐 3D 位置编码、单 token 轨迹 tokenizer、场景流与机器人动作的耦合专家 |
| 结果 | LIBERO 99.8%、LIBERO-Plus 89.1%、RoboTwin Randomized 93.04%、VLABench 56.2%；1.7B 参数 |

## 开源核查（2026-10-12）

- 项目页链接了 GitHub 仓库，但模型 checkpoint 标注 “Coming soon”。
- 仓库 DexForce/UNITAS 仅含 README、assets/overview.png、.gitignore（2 次提交），无安装/运行说明、无权重、无许可证；训练代码与数据集在发布计划中未勾选。结论：待发布（占位仓库）。
