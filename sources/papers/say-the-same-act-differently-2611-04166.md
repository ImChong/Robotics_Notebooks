# Say the Same, Act Differently
> arXiv 论文来源归档（版本 v1）
- **论文身份：** arXiv:2611.04166
- **中文标题：** Say the Same, Act Differently：解释不变、行为却变的 VLA 诊断
- **原文：** https://arxiv.org/abs/2611.04166v1
- **HTML：** https://arxiv.org/html/2611.04166v1
- **PDF：** https://arxiv.org/pdf/2611.04166v1
- **策展：** [多模空间2026.11上半月](../blogs/duomo_embodied_2026_11_first_half.md)
- **代码：** 未核验可运行官方仓库入口
- **核对范围：** 已通读 arXiv:2611.04166v1 §1–§5、主要表 1–3、附录 A.1–A.10 的输入协议、文本/动作读出、离线碰撞定义、跨模型 hook 校验、局限、算力及释放边界。
- **通读：** 已完成
## 原文核验要点
- **研究问题**：视觉语言动作模型相同文字 rationale 可能伴随不同动作；TNAS 用 action Jacobian 与文本 NLL/margin 梯度的正交投影检测此方向。
- **评测**：200 Alpamayo clip、200 InstructVLA 例，离线几何碰撞仅 Alpamayo 154 有障碍标注 clip；非真实道路事故。
- **边界**：附录明确不发布攻击完整管线/优化扰动；论文不能证明系统安全，也不构成防御方法。

## 后续待核
原文已核对完毕。作者实验和方法均为**论文报告**，未独立运行仿真/真机或大型 GPU 训练；代码资源以源码归档记录的实际可访问状态为准。

## 对 wiki 的映射
- [Say the Same, Act Differently：解释不变、行为却变的 VLA 诊断](../../wiki/entities/paper-say-the-same-act-differently-2611-04166.md)
