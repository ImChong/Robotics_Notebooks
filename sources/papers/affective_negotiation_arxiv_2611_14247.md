# Personalized Affective Perception via Continual Learning in Human-Agent Negotiation｜持续学习个性化情感感知的谈判机器人

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14247
- **PDF：** https://arxiv.org/pdf/2611.14247
- **核对范围：** arXiv HTML v1 正文 I–VII：CLIFER 持续学习、效价-唤醒差分反馈、谈判策略、66 人配对试验和统计讨论；核对官方 README 的 verify_all_results.py 与 NAOqi 运行边界；未执行真机。
- **通读：** 已完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/monurkeskin/Personalized-Affective-Perception-via-Continual-Learning-TAFFC-2026

## 核心摘录

1. 从表情预测个性化 valence-arousal 状态。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-affective-negotiation.md)
2. 异步连续学习和生成式回放维护 CLIFER 情感表征。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-affective-negotiation.md)
3. 把情绪方向和强度变化转成目标效用反馈并调节谈判策略。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-affective-negotiation.md)

## 作者报告的结果与限制

66 人配对研究报告代理收益和参与者让步率上升，参与者效用与 Nash 距离无显著改善；无独立表情真值证明情绪识别精度提高。

行为收益不等于准确读懂情绪；部署依赖旧式机器人软件/感知资源且存在伦理边界。

## 原文核对摘录（2026-11-14）

论文 CLIFER 对面部表情的 Valence-Arousal（V,A）预测采用生成式回放，异步更新个体模型；把会话中的归一化情绪变化分解为角度旋转 `atan2(det,dot)` 与向量模长差，形成受限的差分反馈，再改变谈判 Solver 的效用函数。66 人同人配对比较静态 FaceChannel 与持续学习模型，报告代理收益与让步率提高，但参与者效用与 Nash distance 未显著改善。情绪量来自模型**估计值**而非独立标注的情绪真值，因此不能以谈判收益反推识别精度更高。源码支持 `python verify_all_results.py` 复核公开结果；实时机器人流程涉及 Python2.7/NAOqi、FaceChannel 权重及实验室资源，统计脚本不能替代真机运行。

## 复现边界

尚未独立运行数据、训练、评测或机器人试验；论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-affective-negotiation.md)。
