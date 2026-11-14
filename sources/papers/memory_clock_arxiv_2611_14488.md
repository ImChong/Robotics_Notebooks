# Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies｜机器人的记忆可能只是时钟

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14488
- **PDF：** https://arxiv.org/pdf/2611.14488
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、官方源码 README 与公开入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/bennidict23/memory-clock

## 核心摘录

1. 在 RoboMME 计数任务中加入不改变物体或任务状态的暂停和空抓动作。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-memory-clock.md)
2. 对照正常运行、暂停帧进入/不进入记忆、时间戳单独缩放，定位虚假进度来源。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-memory-clock.md)
3. 在测试时按关键事件而非时钟均匀采样记忆；仅删除时间编码不能消除全部捷径。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-memory-clock.md)

## 作者报告的结果与限制

PickXtimes 两次 5 秒暂停后 FrameSamp-Modul 从 47/50 降至 16/50；不把暂停帧写入记忆且停止时间推进时恢复为 45/50。论文摘要称另一任务 44/50→10/50。

事件检测器依赖环境任务信号；RoboMME 仿真结果不能直接代表开放世界真机。

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
