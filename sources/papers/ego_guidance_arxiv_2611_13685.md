# Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos｜无动作标签 Ego 视频辅助机器人策略

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13685
- **PDF：** https://arxiv.org/pdf/2611.13685
- **核对范围：** arXiv HTML v1 全文：§1–5 的 SPAR 方法、数据筛选、仿真/真机评测与附录 A–C 的训练细节及消融。核对官方 README 训练/评测命令；未运行训练。
- **通读：** 已完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/llong-cs/EgoGuidance

## 核心摘录

1. 预训练视觉语言模型上共享机器人动作监督和人类第一视角视觉未来预测。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ego-guidance.md)
2. 训练期使用双分支表征学习并关注视频筛选和分布差异。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ego-guidance.md)
3. 推理时不必运行辅助未来视觉预测分支，仅部署动作策略。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ego-guidance.md)

## 作者报告的结果与限制

摘要报告分布外任务与视频扩充的收益；新闻附件所列五项真机任务共 50 次测试，模型成功 20 次、基线 22 次，不支持所有真机任务均获提升的说法。

无动作标签视频与机器人执行域存在差距，少量真机结果难支持广泛泛化。

## 原文核对摘录（2026-11-14）

论文 §3：Qwen3.5-2B 共享视觉语言骨干，机器人样本通过 flow matching 训练动作专家；机器人与无动作标签第一视角视频的未来视觉特征预测是训练辅助目标，推理时丢弃未来分支。Ego4D 原始约 3976 h，筛光流后 1893 h，再筛可见手后约 975 h。§4–5：LIBERO/LIBERO-Plus 的 OOD 提升不能代表所有真机条件获益。五项 Cobot Magic V2.0 真机任务每项 10 次，Ego Guidance 20/50，π0.5 基线 22/50；在 Cucumber Transfer 灯光扰动 9/10 对 7/10，而 Flower Handover 为 2/10 对 5/10。附录给出阶段配置、不同编码器与视频筛选消融，任务/视觉域依赖仍是局限。

## 复现边界

尚未独立运行训练、推理或机器人试验；论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-ego-guidance.md)。
