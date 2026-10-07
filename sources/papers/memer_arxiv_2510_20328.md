# MemER: Scaling Up Memory for Robotic Control via Experience Retrieval（arXiv:2510.20328）

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2510.20328>
- **项目页：** <https://jen-pan.github.io/memer/>
- **代码：** <https://github.com/memer-policy/memer>（项目页 Code 链接；已公开）
- **机构 / 发表：** 斯坦福大学（Stanford University）；ICLR 2026
- **作者：** Ajay Sridhar、Jennifer Pan、Satvik Sharma、Chelsea Finn
- **一句话说明：** 让高层 VLM 从历史经验中选择并追踪关键帧，再把这些视觉记忆转成语言子任务交给低层 VLA 执行，以支持需要数分钟记忆的长程操作。
- **入库日期：** 2026-10-07

## 核心摘录

1. **问题：** 把长历史全部输入策略计算昂贵且易受分布偏移影响；随机抽帧又会留下冗余和无关内容。
2. **架构：** 高层策略读取任务指令、近期图像与已选关键帧，产生语言子任务和候选历史关键帧；低层策略结合子任务、当前图像和关节状态输出动作。
3. **记忆筛选：** 聚合高层候选关键帧；项目页示例以相邻候选的单链聚类合并距离为 5 帧，再取每簇候选的中位帧写入记忆。
4. **实验：** object search、counting、dust & replace 三类真实长程任务；项目页展示了对比 1、8、32 帧历史，并报告系统能在多次失败抓取和重试后继续完成任务。
5. **实现细节差异：** arXiv 摘要写高层 Qwen2.5-VL-7B-Instruct；项目页当前文字写 Qwen2.5-VL-3B-Instruct。入库时保留这一来源差异，不把版本号混为一谈。

## 对 wiki 的映射

- [paper-memer](../../wiki/entities/paper-memer.md)
- [robot-in-context-learning](../../wiki/concepts/robot-in-context-learning.md)
