# Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies｜机器人的记忆可能只是时钟

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14488
- **PDF：** https://arxiv.org/pdf/2611.14488
- **核对范围：** arXiv HTML v1 正文 §1–7 和附录 A–L 的零事件协议、无时钟消融、事件索引、统计检验；核对官方 README 的 run_eval.sh；未独立运行 GPU 评测。
- **通读：** 已完成
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

## 原文核对摘录（2026-11-14）

论文 §4–5：RoboMME 记忆策略的 32 个时间采样槽位，可能把暂停帧放入记忆并误以为任务进度提高。在 PickXtimes 两次各 5 秒暂停的对照中，47/50（正常）→16/50（暂停帧可见）→45/50（隐藏暂停且冻结索引）。仅删除显式时间戳 NoClock 仍会留下索引层 temporal shortcut。§6 Event-index 使用任务关键接触/状态转换更新记忆，无需重训但部分依赖模拟器信号；普通 pause augmentation 未根治问题。附录 A–L 比较不同任务、事件类型、随机种子，采用成对 McNemar 检验、Wilson 区间及 Holm 校正；有效重复动作可能被误去重，真实时间约束任务仍需要时钟。

## 复现边界

仍未独立运行训练、评测与机器人实验，论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-memory-clock.md)。
