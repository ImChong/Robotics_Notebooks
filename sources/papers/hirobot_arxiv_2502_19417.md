# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models / 分层视觉语言动作模型的开放式指令遵循

> 来源归档（ingest）；复核日期：2026-11-07。

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2502.19417>
- **原文版本：** [arXiv v1，2025-02-26](https://arxiv.org/html/2502.19417v1)
- **正式发表：** [ICML 2025，PMLR 267:54919–54933](https://proceedings.mlr.press/v267/shi25d.html)
- **项目页：** <https://www.pi.website/research/hirobot>
- **项目页归档：** [官方介绍与开放状态](../sites/hirobot.md)
- **机构：** 物理智能（Physical Intelligence）；斯坦福大学（Stanford University）；加州大学伯克利分校（UC Berkeley）
- **第一作者：** Lucy Xiaoyang Shi
- **入库日期：** 2026-09-28
- **核对范围：** v1 全文：摘要、§1–6、参考文献、附录 A 合成数据与 B 系统配置；另交叉核对官方项目页指标与资源链接、PMLR 正式发表记录。
- **通读：** 已完成

## 核心摘录（面向 wiki 编译）

- §4：高层和低层均从 PaliGemma-3B 初始化，分别训练；高层输出语言子任务及可选口头回应，低层 π₀ 用 flow matching 预测动作块。这里是同类骨干，不是共享一套在线权重。
- §4.1–4.3：高层每秒或新用户插话时重算；演示切分成 1–3 秒技能，图像与技能标签用于反向合成可能的提示、插话和回应。
- §5.2：每任务每方法 20 次试验；盲评者分别评估指令准确率 IA 与任务进度 TP。IA 不是整段任务成功率，TP 是物体放置/配置的完成比例。
- §5.4：移除合成数据、或将相同合成数据直接用于扁平 VLA，均弱于完整方案；不能把收益全部归于“多加一个大模型”。
- §6、附录 A/B：高低层分开训练，能力耦合有限；合成数据依赖提示设计。实验按任务分别训练高层，使用 1–2 张 RTX 4090；不是已验证的跨任务统一高层。
- **对 wiki 的映射：** [Hi Robot 主实体](../../wiki/entities/paper-hi-robot.md)

## 开放状态（2026-11-07）

官方项目页未列 Hi Robot 专属训练/推理代码、权重或数据下载。公开论文和视频不等于完整实现开放；[openpi](https://github.com/Physical-Intelligence/openpi) 是低层家族资源，不能代替 Hi Robot 高层与合成数据管线。源码运行时序图不适用。
