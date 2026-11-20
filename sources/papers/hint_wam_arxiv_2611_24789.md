# HINT-WAM｜Hindsight-Informed Training for World Action Models

- **类型：** paper / arXiv:2611.24789
- **原文：** https://arxiv.org/html/2611.24789
- **摘要：** https://arxiv.org/abs/2611.24789
- **项目页：** https://tuandattt.github.io/hintwam/
- **代码：** https://github.com/tuandattt/hintwam_code
- **核查日期：** 2026-11-20
- **核对范围：** arXiv HTML §3 方法三阶段、§4 LIBERO/RoboTwin 与遗忘表、关键消融及附录训练成本；未逐段完成全部附录与引用
- **通读：** 未完成

## 原文记录

1. **问题：** 仅用演示做 SFT，动作生成专家没见过自己的采样中间状态；只给稀疏奖励的 RL 不直接提供动作分布目标。
2. **三阶段：** Stage 0 在演示上训练 action expert；Stage A 冻结基模型，训练可读示范未来视频的 hindsight teacher；Stage B 学生在自身采样去噪状态上受教师分布的 energy score 监督，并加 BC anchor。
3. **推理边界：** 未来视频仅供教师训练使用，部署时学生只读当前观察；视频专家全程冻结。
4. **实验：** LIBERO-90 90 个未见任务、3,921 条示范、每任务 50 次固定初态评测。论文 Table 1：SFT 78.88%，HINT-WAM 85.18%，Full SFT 81.24%。Table 2 无 replay 原任务 4 套件平均 HINT-WAM 44.05%、SFT 37.45%；**遗忘并未消除**。
5. **条件：** 最佳值基于 Fast-WAM，同基座对照；不外推到真实机器人。训练成本与相同评测协议需结合论文附录复核。

**原文未完整通读的边界：** 该资料已核对核心正文与若干附录，但并未逐段校核全部附录及论文引用，因此不标“已完成”。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-hint-wam.md)
