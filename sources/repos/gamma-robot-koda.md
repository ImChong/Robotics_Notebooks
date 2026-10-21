# Gamma-Robot/KODA

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/Gamma-Robot/KODA>
- **项目页：** <https://koda-humanoid.github.io/koda/>
- **论文：** [arXiv:2610.21152](https://arxiv.org/abs/2610.21152)
- **样例与检查点：** <https://huggingface.co/datasets/TeleEmbodied/koda-assets>
- **许可：** README 说明集成代码与 vendored upstream components 各自保留许可；未声明统一的项目级许可，MANO、机器人网格和模型检查点可能另有条款
- **源码状态：** 部分公开；M1/M2 可运行，M4 部分数据公开，M3/M5 计划中（核验日期：2026-10-20）
- **知识节点：** [KODA](../../wiki/entities/paper-koda-egocentric-vla.md)

## 仓库内容与运行入口

当前实现覆盖 RGB-D/MCAP 转换、手部估计与重定向、接触感知手部细化、采样式命令优化跟踪、手臂细化，以及同步的机器人机载视角合成。README 提供 `uv sync --extra cuda` 安装流程；`uv run koda run` 依次运行 `convert`、`track`、`view-change`，可输出处理后的观测、G1 与 Dex3 运动学和合成机载视频。`koda assets download/verify` 管理公开资产。

Hugging Face 数据集目前含两个示例 MCAP 和微调的 ScaleBFM tracking checkpoint。M3 控制器动作标签、motion-token 编码与控制器对齐数据导出，M5 VLA 训练配置与模型细节不在当前发布中；这不是完整论文管线的可复现实现。

## 关联资料

- [项目页归档](../sites/koda-humanoid.md)
- [论文摘录](../papers/koda_arxiv_2610_21152.md)
