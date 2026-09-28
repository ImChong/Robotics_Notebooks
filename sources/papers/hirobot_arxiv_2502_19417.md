# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models（arXiv:2502.19417）

> 来源归档（ingest）

- **标题：** Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
- **短名：** Hi Robot
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2502.19417>
- **项目页：** <https://www.pi.website/research/hirobot>
- **机构：** 物理智能（Physical Intelligence）；斯坦福大学（Stanford）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-09-28
- **一句话说明：** 高层 VLM 把开放指令和现场反馈说成短语言步骤，再交给 π₀ 低层执行。

## 开源状态（步骤 2.5，2026-09-28）

- **确认未开源**：项目页与 arXiv 摘要页未列 GitHub、权重或数据集。低层策略所依托的 [openpi](https://github.com/Physical-Intelligence/openpi) 不是 Hi Robot 训练与合成标注代码。

## 核心摘录（面向 wiki 编译）

- 同一 VLM 骨干分两级：高层产出下一步语言命令，低层 π₀ 输出动作；用户中途纠正（如 “that's not trash”）由高层重新接地后再交给低层。
- 训练不只靠原子技能标注，还用合成的复杂提示和人工插话，把已有演示配对成多轮交互。
- 博客图给出桌面清理、做三明治、购物三类的指令跟随：Hi Robot 平均指令准确率 76，扁平 VLA 为 36；并称相对 GPT-4o 的指令跟随准确率高 40%。这些是作者任务，不是公共榜。
- **对 wiki 的映射：** [paper-hi-robot](../../wiki/entities/paper-hi-robot.md)
