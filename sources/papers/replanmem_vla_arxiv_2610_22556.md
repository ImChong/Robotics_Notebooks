# When Plans Change, Remember: Replanning-Guided Visual Memory for Vision-Language-Action Models（官方论文）

> 来源归档（ingest）

- **标题：** When Plans Change, Remember: Replanning-Guided Visual Memory for Vision-Language-Action Models
- **方法名：** RePlanMem-VLA（Replanning-Guided Keyframe Memory for VLA）
- **类型：** paper / preprint
- **作者：** Victoria Smelova、Daniil Kazachkov、Georgiy Kazantsev、Iuliia Tsvetkova、Egor Cherepanov、Nikita Kachaev、Artem Latyshev、Aleksandr I. Panov、Alexey K. Kovalev
- **机构：** MIRAI、Independent Researcher、Central University、AXXX、CogAI Lab（项目页所列）
- **论文：** <https://arxiv.org/abs/2610.22556>
- **项目页：** <https://vika787.github.io/RePlanMem/> — [项目页归档](../sites/replanmem-project.md)
- **代码：** 截至 2026-10-20，项目页未列公开代码仓库。
- **入库日期：** 2026-10-20

## 核心内容

长程任务中的 VLA 可能需要稍后使用当前视野中已消失的物体属性或任务阶段信息。RePlanMem-VLA 将滚动时域执行中相邻 action chunks 对同一重叠未来动作的预测分歧，作为策略内部的事件 / 关键帧写入信号；随后将选中的观察保存到有界视觉记忆，并通过 robot-masked filter 抑制重复帧。

论文先在从头训练的自回归 decoder-only 与 Flow-Matching Transformer 上比较 35 种分歧指标，再在 OpenVLA-OFT 和 π₀.₅ 上验证选定信号。摘要报告事件检测 ROC-AUC 为 0.981–0.997；在论文评估的记忆保留实验中，相关观察在 100% episode 中保留至后续决策点。真机部分使用 replay data 验证事件选择和去冗余行为，不等价于完整的在线真机策略部署评测。

## 对 wiki 的映射

- [RePlanMem-VLA 论文实体页](../../wiki/entities/paper-replanmem-vla.md) — 以 action-plan revision 触发有界视觉记忆。
- 关联 [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) 与 [KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) 的关键帧记忆路线。
