# A Fix for Softmax and Cross-Entropy（arXiv:2610.31642）

> 来源归档（ingest · 依据 arXiv v1）

- **标题：** A Fix for Softmax and Cross-Entropy
- **类型：** paper / numerical stability / softmax / cross-entropy / knowledge distillation
- **arXiv：** <https://arxiv.org/abs/2610.31642>（v1，2026-10-22；HTML：<https://arxiv.org/html/2610.31642>；PDF：<https://arxiv.org/pdf/2610.31642>）
- **作者：** Jonathan T. Barron、Dan Gnanapragasam、Michael Tschannen
- **机构：** Google DeepMind
- **许可：** arXiv HTML 标注 CC BY 4.0
- **入库日期：** 2026-10-25
- **一句话说明：** 解释标准 softmax / log-softmax / cross-entropy 的两类浮点精度问题，并给出稳定的 hard-target 与 soft-target 计算：避免高置信度下目标梯度消失，以及蒸馏中学生与教师 logits 相等时仍产生伪梯度。

## 摘录 1：两类数值问题

1. **Softmax Collapse（高置信度梯度坍缩）：** 稳定 softmax 通常将最大 logit 移至 0 后求指数和。若正确类别 logit 远大于其他 logits，分母会计算为 \(1+e^{-m}\)。当 \(e^{-m}\) 小到加到 1 后被舍入掉时，概率成为精确的 1，标准交叉熵的目标 logit 梯度 \(q_t-1\) 也变为 0；非目标类别的小正梯度仍存在，梯度和因此偏离 0。论文报告的精度悬崖约为 bf16 的 \(m=5.5\)、fp32 的 \(m=16.6\)、fp64 的 \(m=36.7\)。

2. **Softmax Cancellation（软目标相消）：** 知识蒸馏中，标准梯度通过分别计算教师概率 \(p\) 和学生概率 \(q\)，再求 \(q-p\)。当两组 logits 很接近时，近似相等概率相减会丢失精度；即使学生与教师 logits 完全相同，前向 softmax 与反向 log-softmax 的运算差异也可能留下非零梯度。

## 摘录 2：修正方法

- **硬标签路径：** 从 log-sum-exp 中排除最大 logit或目标 logit，直接求 margin \(m\)，再用稳定的 \`softplus(-m)\` 表达损失。目标梯度写成 \(-e^{-m}/(1+e^{-m})\)，避免先计算接近 1 的概率再减 1。论文给出 JAX 的 log-softmax、softmax 和 hard-target cross-entropy 实现。
- **软标签路径：** 直接使用学生与教师 logits 的差，在 log 空间计算概率比；使用 \`expm1\`、\`log1p\` 保留接近零的差值，并校正残余浮点漂移。论文的 JAX 自定义梯度在两者 logits 相等时为零。

## 摘录 3：论文报告的结果

- 在模加法 grokking 基准 \(a+b\bmod 31\) 上，使用 weight decay、AdamW 与 fp32 时，标准交叉熵会在精度悬崖附近出现 Slingshot 损失尖峰；修正版在 5 个随机种子上都更快完成 grokking。
- 对 \(a^2+b^2\bmod 97\)（无 weight decay）的设置，论文报告标准交叉熵的一次尖峰使测试准确率从 61% 降至 23%；修正版继续训练到 85%。
- 在 4 个 grokking 任务上，fp32 修正版匹配论文中的 fp64 标准交叉熵表现，并达到 100% 测试准确率。
- 在模加法 \(a+b\bmod 31\) 的表格对照中，修正版在 fp32、bf16、fp64 及多种 Adam \(\epsilon\) 设置下达到 98.9–100% 测试准确率。
- 蒸馏实验将学生初始化为教师权重。标准软目标交叉熵在 bf16 下的残余梯度会推动学生偏离教师；修正版在各测试精度下保持零 KL 与 100% 测试准确率。

以上为论文在小规模合成 grokking / 蒸馏任务中的实验结论，不是机器人策略训练的直接验证。

## 对机器人学习的工程关联

策略网络常使用交叉熵类目标训练离散动作、动作 token 或蒸馏目标。论文提示：若训练日志在拟合后 loss 归零、梯度和不再近似为零，或蒸馏学生无故偏离教师，可检查 logit margin、精度格式及 soft-target 梯度。是否影响某个机器人训练流程，仍需在该任务、框架和硬件上验证。

## 对 wiki 的映射

- [论文实体页](../../wiki/entities/paper-softmax-cross-entropy-fix.md)
- [Kaplan Scaling Laws](../../wiki/entities/paper-scaling-laws-neural-language-models.md) — cross-entropy 作为训练损失的相关背景
- [AWR](../../wiki/methods/awr.md) — 加权交叉熵策略更新的机器人学习实例
