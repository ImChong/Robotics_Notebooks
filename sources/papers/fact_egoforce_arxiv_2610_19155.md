# FACT: A Multimodal Force–Action–Track Foundation Model for Human-to-Robot Dexterity（arXiv:2610.19155）

> 来源归档（ingest）

- **标题：** FACT: A Multimodal Force–Action–Track Foundation Model for Human-to-Robot Dexterity
- **类型：** paper / dexterous-manipulation / tactile / human-to-robot-transfer / flow-matching
- **arXiv：** <https://arxiv.org/abs/2610.19155>（v1，2026-10-13）
- **HTML / PDF：** <https://arxiv.org/html/2610.19155v1> · <https://arxiv.org/pdf/2610.19155>
- **项目页：** <https://fact-1-0.github.io/> — 论文列出的作者网站；归档见 [项目页来源](../sites/fact-1-0-github-io.md)
- **代码与数据：** 截至 2026-11-08，作者项目页标注代码和数据即将发布，未列权重入口；见[项目页来源](../sites/fact-1-0-github-io.md)
- **核对范围：** 已逐节核对 arXiv HTML v1 全文：摘要、I–VI 节、表 I–IV、附录 A–E（含采集硬件、力标签、推理与对照基线设置）；以论文为一手研究依据。
- **通读：** 已完成
- **复核日期：** 2026-11-08
- **机构：** 密歇根大学（University of Michigan）、斯坦福大学（Stanford University）、微软研究院（Microsoft Research）
- **作者：** Zhi Cao、Yining Hong、Jiaolong Yang、Yejin Choi、Li Fei-Fei、Jiajun Wu、Huang Huang
- **入库日期：** 2026-10-16
- **一句话说明：** 使用 EgoForce 人类触觉示范预训练；视觉轨迹引导粗动作规划，再联合预测未来力并精修动作，迁移到真实双臂灵巧操作。

## 摘要级要点

- **问题：** 大规模人类操作视频通常没有接触力标注；机器人灵巧操作演示采集昂贵。
- **数据：** EgoForce 用头戴与腕部三视角、双手触觉手套和自动力校准，采集约 2,000 小时、40,000 个 episode、1,000 项任务、1,000 名演示者。
- **方法：** Track-Guided Coarse Planner 联合预测未来视频、视觉点轨迹和动作块；Force–Action Joint Refiner 使用当前触觉历史，联合预测未来指尖力并精修剩余动作。
- **评测：** 十项真实双臂任务，每项任务用 10 条机器人轨迹与 90 条人类示范微调。摘要报告相对 T-Rex 的任务级最高提升 60 个百分点。
- **边界：** 数据规模不代表数据已公开可下载；小样本实机结果不能直接外推为部署可靠性。

## 训练、部署与实验细节（原文核对）

- **预训练**：EgoForce 约 2,000 h 人类演示 + T-Rex 约 100 h 机器人遥操作，按时长比例抽样。使用 240 张阿里巴巴平头哥 PPU（Zhenwu 810E、96 GB）；粗规划器约 5.05B 参数，联合精修器约 65M 参数。论文没有给出可下载 checkpoint。
- **数据与动作**：演示按 LeRobot-v2 格式整理，动作/点轨迹 20 Hz、视频 10 Hz、触觉力 5 Hz；16 步动作块共 0.8 秒。每条动作 62 维：双臂各 3D 位移 + 6D 旋转 + 22 个灵巧手关节目标。
- **采集与力标签**：FusionX 头部双目 + 双腕摄像头 + TouchTronix 触觉手套同步采集；人类手指压力阵列仅能近似法向力及两个力矩，切向力和扭转力矩置零；不是人类手部完整 6 轴实测力。
- **推理**：Coarse Planner 先做 6 步 Euler 流匹配去噪，停于 τ=0.4，提取视觉预测与 KV 缓存；Refiner 用 4 步 Euler 联合细化剩余动作和未来力。整块 20 Hz 执行，指尖触觉每 5 Hz 刷新（块内偏移 4/8/12），不重新运行粗规划器。
- **未来力角色**：模型预测未来力用于辅助动作去噪；它不是要硬件跟踪的参考力，也未展示独立 1 kHz 力矩闭环。
- **微调对照**：每项 10 条机器人示范与另行采集的 90 条人类示范（不是直接抽样 EgoForce），在 8 GPU 上针对任务微调；π0.5 与 T-Rex 也从各自公开底座微调，架构和动作编码仍有不同。
- **指标**：7 项二值任务每项 10 次实机 rollout；2 项乐器任务为 10 个音符中的正确数量；移液器调节需同时看 10 次中成功拨动次数和拨动成功试次的 μL 平均绝对误差。
- **宣传表述核查**：「机器人小脑」「世界首个」「兼容任何大脑」是类比或未经本论文实验验证的推广外延；论文展示的是特定双臂灵巧操作平台上经微调后的效果。

## 对 wiki 的映射

- [FACT（EgoForce · 力–动作联合精修）](../../wiki/entities/paper-fact-egoforce-human-dexterity.md)
- 同名缩写的另一项工作：[FACT（Failure-Aware Causal Training，arXiv:2608.10232）](../../wiki/entities/paper-fact.md)
- 对照阅读：[VT-WAM（视觉–触觉–动作联合模型）](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)

## 核心摘录

1. EgoForce 将校准力信号与视觉、手部动作同步采集；消融显示缺少人类预训练力监督时，部分接触任务明显退化。
2. 粗规划器联合预测未来视频、点轨迹和动作；精修器共同去噪剩余动作和未来指尖力，未来力是预测目标而非控制参考。
3. 七项二值任务各做 10 次 rollout；乐器任务按正确音符数计分；注射器旋钮同时报告参与次数与参与试次的读数误差。
4. 人类手套只能从压力阵列估计法向力及两个力矩分量；切向力和扭转力矩置零。微调所用人类示范另针对评测任务采集。
