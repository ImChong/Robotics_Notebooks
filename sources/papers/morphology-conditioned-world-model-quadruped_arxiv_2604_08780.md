# Morphology-Conditioned World Model for Cross-Embodiment Quadrupedal Locomotion（arXiv:2604.08780）

> 来源归档（paper）

- **标题：** Morphology-Conditioned World Model for Cross-Embodiment Quadrupedal Locomotion（v1 标题为 *Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning*）
- **类型：** paper
- **作者：** Mohamad H. Danesh、Chenhao Li、Amin Abyaneh、Anas Houssaini、Kirsty Ellis、Glen Berseth、Marco Hutter、Hsiu-Chin Lin（McGill University / Mila / 蒙特利尔大学 / ETH Zurich）
- **arXiv：** <https://arxiv.org/abs/2604.08780>（v1 2026-04-09，v2 2026-08-17）
- **PDF：** <https://arxiv.org/pdf/2604.08780v2>
- **代码：** 论文未给出 QWM 自身的代码仓库；仅在参考文献引用其异构仿真基础设施 Hetero-Isaac（<https://github.com/modanesh/Hetero-IsaacLab>，非 QWM 实现）
- **入库日期：** 2026-11-12
- **一句话说明：** 在 DreamerV3 的 RSSM 上加入由机器人描述文件（USD）提取的形态向量 µ，并配合分机器人类型的奖励归一化，让同一个世界模型与策略零样本控制留出的四足机器人。
- **核对范围：** 读了 v2（2026-08-17）全文：摘要、第 1–7 节、参考文献，以及附录 A–H 的文字、表格和图注（环境与奖励、网络结构、形态多样性与留一偏差分析、消融、真机定量评测、潜状态解耦、µ 扰动与未建模动力学鲁棒性）。同时读了 v1（2026-04-09）全文，用于核对两个版本的差异。图只读了图注和文本提取出的数值，没有逐图目视；v2 附录 G 的 t-SNE 细节和附录 H.3 实现细节只读了结论性段落。未读论文引用的其他文献，未查看真机视频。
- **通读：** 已完成

## 开源状态

- 论文与 arXiv 页未给出 QWM 的代码或项目页链接；已核对全文，唯一的仓库链接是参考文献 [14] 的 Hetero-Isaac（作者自建的异构 Isaac Lab 仿真环境），不是 QWM 训练代码。结论：QWM 本身未见开源（以论文文本为准）。

## 核心摘录

1. **问题（v2 摘要）：** 世界模型的动力学通常"形态锁定"；论文比较两条用法：把形态说明喂给无模型策略，或喂给学习到的动力学模型再在想象中提取策略，并主张第二条。
2. **方法（第 4 节）：** 在 DreamerV3 的 RSSM 上做三件事：形态条件世界模型（双塔编码器加循环状态每步注入 µ）、物理形态编码器 PME（从描述文件取运动学、几何、质量、驱动四组特征并 min-max 归一化到 [-1,1]）、自适应奖励归一化 ARN（按机器人类型维护回报 5%–95% 分位差的 EMA）。µ 假设四足三段腿模板，12 维动作。
3. **设置（第 5 节、附录 A）：** Isaac Lab 上的异构环境，8 台机器人：ANYmal B/C/D、Unitree A1/Go1/Go2/B2、Spot；4096 并行环境，回合最长 1000 步；训练 36M 环境步，5 个随机种子；任务是平地盲走速度跟踪。
4. **训练集表现（第 5.1 节）：** PME-PPO 与 QWM 渐近性能相当，PME-PPO 约需两倍环境步才开始收敛；DreamerV3、PWM、TWISTER 不给 µ 时未达到可控策略。
5. **零样本（第 5.3 节、表 1）：** 每个目标机器人单独在其余 7 台上从头训练再留出评测。v2 表 1：ANYmal-D 奖励 18.2、回合长 948.6；Go1 35.5、974.4；B2 12.1、925.4；专家 PPO 上界分别 21.8/981.3、39.7/996.1、13.6/961.1；PME-PPO 为 10.1/530、23.1/602、-0.2/337。
6. **真机（第 5.4 节、表 2）：** 留出的 Go1 与 ANYmal-D，每台 10 次×60 秒，平地室内，随机速度指令，无跌倒；线速度误差 QWM 0.34/0.30 对专家 PPO 0.31/0.28（Go1/ANYmal-D）。
7. **鲁棒性（附录 H）：** 对 µ 乘性扰动 20% 时 QWM 保留约 80% 回合长，PME-PPO 约三分之一；扭矩衰减和载荷下 QWM 保留 75–84%。
8. **作者自述的局限（第 6 节）：** µ 只覆盖固定的四足三段腿模板，12 自由度；评测限于盲走平地速度跟踪。

## v1 与 v2 的差异（本次核对发现）

- B2 留出结果：v1 表 I 为 QWM 奖励 2.2、回合长 405.2，作者承认"B2 外推失败"；v2 表 1 为 12.1、925.4，称接近专家。v2 把"距离"分析改成逐特征留一 z-score，并称 B2 是"组合式缺口"。两版在训练协议或评测细节上的差别论文未逐项说明。
- 其他留出结果也有变化：ANYmal-D 奖励 15.2 变 18.2，Go1 30.5 变 35.5。
- 开环预测视野：v1 为 45 步，v2 正文为 85 步（附录 H 仍用 45 步）。
- 真机摘要：v1 称"约 10% 内"，v2 正文改为"约 10–13%"。

## 对 wiki 的映射

- [paper-morphology-conditioned-world-model-quadruped](../../wiki/entities/paper-morphology-conditioned-world-model-quadruped.md)
