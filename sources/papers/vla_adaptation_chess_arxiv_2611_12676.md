# VLA Adaptation with Real and Simulated Data: A Case Study on Robotic Chess
- **原文：** https://arxiv.org/html/2611.12676v1
- **论文编号：** arXiv:2611.12676v1
- **作者：** Matěj Kripner、Milan Straka（Charles University）
- **官方代码：** https://github.com/Kripner/vla-chess
- **真实示范集：** https://huggingface.co/datasets/Kripi/so101-chess-king-moves-real
- **日期：** 2026-11-11（本次核对）
- **核对范围：** arXiv HTML 全文、方法和训练协议、仿真实验、实机适配与消融；对照 README、仿真采集脚本、评估脚本和微调入口。
- **通读：** 已完成

## 核心摘录
1. 指令为 `Move the king from e4 to a5` 一类棋盘坐标搬运：SO-101 机械臂抓起**单个棋王**并移动到目标格，不是棋类规则理解或通用对弈。
2. 提供 MuJoCo 数字孪生（棋盘、棋子、SO-101、两相机、基于实际标定参数）、脚本专家、LeRobot 格式数据、固定测试 schedule、π0.5/π0/SmolVLA 训练与评测配方。
3. 官方提供真实遥操作 261 条棋王搬运轨迹，带两路视频、关节状态/动作、阶段标注与 outcomes；论文具体训练样本数的过滤条件和任务成功定义应与 `splits/` 及评测脚本核对。
4. 大规模合成示范并非自动保证性能：在论文报告的 10k 示范量级上，训练更新数与学习率 cooldown 会显著改变成功率。论文展示同样 trunk checkpoint 下 10% 冷却阶段后的成功率可远高于未经冷却版本。
5. 真实适配中仿真预训练 + 128 条真实示范约 18/25 成功，相对于仅真实数据基线 2/25；但 25-trial 样本限制不可忽略。局部 reward/on-policy 继续训练尝试并非普适获益。
6. 按官方仓库，评估含 `success`（精确放置）与 `lenient`（落在目标格），不得把两种口径混为一个比例。

## 开源边界
- [源码核验](../repos/vla-chess-official.md)：官方代码 Apache-2.0；可公开获取 MuJoCo 场景、采集、微调、闭环评估及真实演示数据，但训练权重和真机伺服配置仍需用户环境支持。
- **实验结果：** 作者自报，本次未在 SO-101 上复现实验。
- [主实体](../../wiki/entities/paper-vla-adaptation-chess.md)
