# XPACE（arXiv:2609.17372）

> 来源归档（paper）

- **标题：** XPACE: Joint World and Action Modeling from Heterogeneous Experience
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.17372>
- **PDF：** <https://arxiv.org/pdf/2609.17372> · [项目页 PDF](https://xpeng-robotics.github.io/xpace/assets/papers/xpace.pdf)
- **项目页：** <https://xpeng-robotics.github.io/xpace/>
- **机构：** XPENG Robotics（小鹏机器人）
- **入库日期：** 2026-09-20
- **一句话说明：** 统一 WAM + world simulator：共享 video backbone 从异构经验联合学视频与动作，simulator 合成 deviation–recovery 闭环提升 policy；IRON 人形真机验证人–机技能迁移与自改进增益。

## 开源状态

- **未开源**（步骤 2.5 核查，2026-09-20）：项目页仅链至 [xpeng-robotics](https://github.com/xpeng-robotics) 组织，**无 XPACE 专用仓库或公开权重**；详见 [`sources/sites/xpace-project.md`](../sites/xpace-project.md)。

## Abstract（arXiv）

通用机器人需要从多样经验中选动作并预见动作如何改变世界。XPACE 是统一具身世界模型：既是 **world action model**（联合预测可执行动作与未来视频），也是 **world simulator**（给定 prescribed action 预测视觉后果）。核心洞见：视频预测既能把异构经验接到动作学习，也能为 policy 生成新经验。policy 与 simulator 共享 video backbone；无动作视频学视觉动力学，人/机带动作示范联合学视频与动作。粗到细课程逐步强调机器人控制并保留人经验，使 policy 学到超出机器人示范的行为。此外 simulator 适配自生成上下文，围绕专家示范合成 deviation–recovery 轨迹，在过滤后的 recovery 样本上微调 policy。XPENG **IRON** 人形实验表明：异构训练提升鲁棒性并支持人观察到、但机器人示范缺失的技能迁移；simulator 生成的 recovery 数据进一步提升真机任务完成率。

## 核心摘录

1. **双模式统一架构：** 共享 causal Video Transformer；policy 模式用历史 + 语言指令预测未来视频与 16-step action chunk；simulation 模式用 skeleton control + camera pose 预测视频，action 分支关闭。
2. **异构数据金字塔（约 5000h）：** L1 无动作 egocentric 视频、L2 人视频–动作、L3 task/appearance 对齐 bridge、L4 IRON 遥操作；失败/恢复集 **只训 simulator、不进 policy 模仿**。
3. **三阶段训练 + 后训练分叉：** Stage I 无动作视频适配预训练 video backbone；Stage II flow-matching 联合训练（视频–动作 / 仿真等概率），human→bridge→robot 三相位；Stage III 复制 checkpoint 为 simulator（SGF 自梯度 forcing）与 policy（8% 合成 recovery + DAgger 式微调）两支。
4. **真机 headline（项目页，20 trials/task）：** 香蕉 pick-place / 倒水 / 叠碗平均成功率 **68.3%**（DreamZero 40.0%，GR00T 6.7%）；recovery 微调后 **61.7%→86.7%**；叠碗 **不在 robot demo 中** 但 human/bridge 有覆盖。
5. **仿真侧：** skeleton 条件 **token addition** 优于 AdaLN/cross-attn；SGF 在 97/193 帧 rollout 上 PSNR 与 latency 双改善；recovery simulator 为 Phase II-c + SGF 固定 checkpoint。

**对 wiki 的映射**

- [paper-xpace](../../wiki/entities/paper-xpace.md)
- [xpace-project.md](../sites/xpace-project.md) — 项目页定量表与 demo 索引
- [world-action-models](../../wiki/concepts/world-action-models.md) — WAM 概念族谱
- [generative-world-models](../../wiki/methods/generative-world-models.md) — 生成式世界模型方法层
