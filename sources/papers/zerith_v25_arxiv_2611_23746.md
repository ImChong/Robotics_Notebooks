# ZERITH 2.5：From Predictive Power to Policy Learnability（视频预测先验如何改善策略可学习性）

> 论文一手资料归档；研究结论需与具体评测条件一起引用。

- **标题：** ZERITH 2.5: From Predictive Power to Policy Learnability
- **类型：** paper
- **论文：** <https://arxiv.org/abs/2611.23746>
- **全文 HTML：** <https://arxiv.org/html/2611.23746v1>
- **官方技术报告：** <https://www.zerith.com/blog/zerith-2-5/assets/ZERITH-V2.5-v1.0.pdf>
- **机构：** Zerith Robotics（ZERITH Z-Lab）
- **代码：** <https://github.com/ZERITH-ZLab/ZERITH-V2.5>
- **研究页：** <https://www.zerith.com/blog/zerith-2-5/>
- **入库日期：** 2026-11-19
- **官方博客复核：** 2026-11-20
- **版本：** arXiv:2611.23746v1；官方技术报告 v1.0
- **核对范围：** 原文方法（Wan 视频 DiT、K/V 注意力、VGGT、S1/S2 目标与噪声、动作梯度调度）、受控对照与权重迁移/梯度消融、完整模型仿真和 H1 真机评测、讨论和附录实施细节；并核对项目 README 与实现入口。原文结果属于作者报告，未在本地复训。
- **通读：** 已完成
- **一句话：** 固定示范与 30k 训练更新预算，先用目标域机器人视频对预测分支做宽噪声视觉适配，再联合学习动作，优于直接联训；增益随持续保留的适配视频权重转移，而不是只靠复制其初始化到动作分支。
- **Wiki 映射：** [ZERITH 2.5 独立实体](../../wiki/entities/paper-zerith-v25.md)

## 1. 为什么做这项研究

视频模型具有强预测能力，却不意味着自然形成可用控制策略。论文研究「哪些视频预测训练会改善下游动作策略的可学习性」，保持机器人示范和总训练步数相同，比较从零直接联合训练与先适配视觉模型再联合训练。

- **模型：** Wan2.2-TI2V-5B 视频分支（30 个 Transformer blocks）+ 流匹配动作策略分支（10 blocks），约 6.774B 可训练参数；接受多视角 RGB、语言指令、机器人状态与 VGGT 几何条件。
- **融合：** 动作分支从视频模型指定层读取 K/V、与几何及自身动作 K/V 做注意力融合；并非先生成完整未来 RGB 视频再经逆动力学得到动作。
- **闭环：** 输出 16 步动作块，在官方仿真服务中返回并执行前四步后重新观测、重新生成。部署时不需要解码未来视频。

## 2. 两阶段学习与控制变量

1. **S1 视觉适配（3,000 updates）：** 干净当前帧作为条件、未来四个视频 latent 槽在较宽噪声区间采样，以视频 flow matching 目标适配目标机器人域。训练的噪声范围比部署纯噪声起点更宽。
2. **S2 联合学习（27,000 updates）：** 视频分支保留 S1 适配权重 W1，策略分支从原始 Wan 权重 R0 初始化；同时优化视频目标与动作流匹配损失。动作→视频梯度在 S2 前 5% 关闭，5%–10% 线性开启，之后全量传递。
3. **Direct 对照（30,000 updates）：** 相同示范与总更新数，跳过 S1 直接进行联合视频–动作训练。

**关键不是总更新数变多，而是视频预测的目标域适配发生在动作联训之前；**后续视频模型仍作为动作分支可读取的视觉特征源。

## 3. 受控实验：严格区分数字口径

| 实验口径 | 论文报告 | 正确解读 |
|---|---|---|
| 图中 Training efficiency 的单次 run | Direct **37.54%**；Staged **56.70%** | LIBERO-Plus；合计 30k 更新，不是完整模型排行榜 |
| Weight transfer 单次对照 | W0/R0 **37.54**；W0/R1 **33.60**；W1/R0 **57.21**；W1/R1 **56.70** | 适配视频权重 W1 所在分支是主要迁移来源 |
| 三种几何条件 × 三种子对照 | 静态 **37.64→56.80**（**+19.16±0.26 pp**）；加法 **+17.93±0.27 pp**；乘法 **+17.55±0.27 pp** | 三种子均值与配对差异样本 SD，不等于上面的单次 run |
| S1 噪声监督消融 | 宽噪声 **56.70**；仅部署纯噪声 **35.90** | 视觉适配的监督分布决定效果 |
| 图中 Gradient preservation | Action→Video **96.0%**；Visual loss **2.9%**；Video-model gradient **4.8%**；权重位移 **1.675%** | 机制分析和关联证据，不是严格因果辨识 |

## 4. 完整模型的评测与边界

**下列指标是完整系统经过机器人预训练等步骤后的结果，不得与受控消融的 56.70% 混用。**

| 仿真 | ZERITH 2.5 成功率 |
|---|---:|
| LIBERO | **98.95%** |
| LIBERO-Plus | **93.70%** |
| MetaWorld MT50 | **92.3%** |

H1 真机 Table 6：Pick-and-place **97/100=97%**，Bin lining **8/20=40%**，Bimanual assembly **2/20=10%**。SKU sorting 是定性机器人演示，不存在可引用的统计成功率。完整版预训练共约 **10,000 h** 机器人语料，其中 **6,000 h** 由 ZERITH H1 自采、另约 **4,000 h** 来自公开语料；不能认作全量数据已开放。

跨模型结果中部分基线使用不同协议，特别是 MetaWorld 基线困难组等权平均与本模型 episode 汇总口径不同。主要因果对照发生在一个视频模型和 LIBERO 家族，结果不能外推到所有 WAM/VLA。

## 5. 推理时延与适用范围

论文 §4.5 在**一张 H800、batch=1** 条件下报告单次决策平均 **331.25 ms**、P95 **365.30 ms**，对应 **3.02 plans/s**；每规划执行四个动作步，折合 **12.08 action steps/s**。这只是执行动作块的有效输出频率，并非底层伺服控制环频率。

分阶段训练也不是所有短预算都更优：5,000 总更新时 Staged **4.22%**、Direct **21.28%**；高预算交点只由有限检查点支持。可学习性结论目前基于 Wan2.2-TI2V-5B、S1 3k updates 和 LIBERO 系列，作者将跨骨干/跨任务推广列为未验证方向。

## 6. 知识库映射

- [ZERITH 2.5 详情页](../../wiki/entities/paper-zerith-v25.md)：将视频适配、K/V 读出与源码推理链路串起来。
- [World Action Models](../../wiki/concepts/world-action-models.md)：判断训练期视觉预测与部署期显式未来生成的边界。
- [OpenWAM](../../wiki/entities/paper-openwam.md)：对照 Joint WAM 研究中的预训练数据配方和架构变量。

## 参考一手资料

- [论文 HTML：arXiv:2611.23746v1](https://arxiv.org/html/2611.23746v1)
- [作者技术报告](https://www.zerith.com/blog/zerith-2-5/assets/ZERITH-V2.5-v1.0.pdf)
- [ZERITH Z-Lab 官方博客](https://www.zerith.com/blog/zerith-2-5/)
- [官方代码、数据和权重说明](https://github.com/ZERITH-ZLab/ZERITH-V2.5)
