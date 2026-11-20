# HINT-WAM｜Hindsight-Informed Training for World Action Models

- **类型：** paper / arXiv:2611.24789
- **原文：** https://arxiv.org/html/2611.24789
- **摘要：** https://arxiv.org/abs/2611.24789
- **项目页：** https://tuandattt.github.io/hintwam/
- **代码：** https://github.com/tuandattt/hintwam_code
- **核查日期：** 2026-11-20
- **核对范围：** arXiv:2611.24789v1 HTML 完整正文（§1–5）、参考文献索引、附录 A–L（训练配置/成本、教师验证、配对统计、任务拆分、RoboTwin、LIBERO 协议、self-imitation、损失/超参选择、toy mode-averaging、rollout、讨论）；对照官方训练 README
- **通读：** 已完成

## 原文记录

1. **问题：** 仅用演示做 SFT，动作生成专家没见过自己的采样中间状态；只给稀疏奖励的 RL 不直接提供动作分布目标。
2. **三阶段：** Stage 0 在演示上训练 action expert；Stage A 冻结基模型，训练可读示范未来视频的 hindsight teacher；Stage B 学生在自身采样去噪状态上受教师分布的 energy score 监督，并加 BC anchor。
3. **推理边界：** 未来视频仅供教师训练使用，部署时学生只读当前观察；视频专家全程冻结。
4. **实验：** LIBERO-90 90 个未见任务、3,921 条示范、每任务 50 次固定初态评测。论文 Table 1：SFT 78.88%，HINT-WAM 85.18%，Full SFT 81.24%。Table 2 无 replay 原任务 4 套件平均 HINT-WAM 44.05%、SFT 37.45%；**遗忘并未消除**。
5. **条件：** 最佳值基于 Fast-WAM，同基座对照；不外推到真实机器人。训练成本与相同评测协议需结合论文附录复核。

**附录核查补充：** 附录 B 表 4：HINT-WAM 完整后训练流水线 71.6 h / 572 GPU-h（8× A5000），SFT 55.8 h / 447 GPU-h，Full SFT 127 h / 1,016 GPU-h；比较时不能混淆训练流水线与单阶段计时。附录 C 表 5：future condition 可使 held-out action loss 下降 3.28%（LIBERO-90），因此 teacher 的 hindsight 必须有实际效力；在不匹配的基础 checkpoint 构建 teacher 反而会伤害学生。附录 D 表 6：HINT-WAM vs SFT 平均 +6.30 个百分点，90 任务配对 t=3.58，但符号检验 p=0.087，不能笼统称每项任务显著获益；附录 E 的 basket/caddy 任务存在负增益。附录 I/J 解释 K=1、能量分数保留多模态而 velocity MSE 易平均化。附录 L 明确 **on-policy 仅指 sampler 的去噪中间状态，不是环境闭环 rollout**。论文统计来自作者，未独立复现实验。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-hint-wam.md)
