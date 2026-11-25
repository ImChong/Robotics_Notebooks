# 神经网络循环简史｜The History of Looping

- **类型：** blog / guide（二手技术导读，不是论文或新模型）
- **原文：** <https://www.alphaxiv.org/guides/history-of-looping>
- **作者：** Kaden Zheng（alphaXiv）
- **页面更新时间：** 2026-11-24（原页标记 Updated）
- **入库日期：** 2026-11-25
- **核对范围：** 原页导言、七个正文小节、五幅图的图注、表 1、侧栏时间线。代表性论文核对摘要与首发日期；未将全部引文逐篇通读。
- **通读：** 已完成

## 导读要点

- 原文用视觉皮层反馈开场；Hupé 等的实验是对特定视觉区反馈的研究，不是 Transformer 计算的生物机制证明。1943 年 McCulloch–Pitts 循环网络用反馈表达历史依赖；1982 年 Hopfield 网络以对称连接和能量下降说明联想记忆。现代连续 Hopfield 更新可与点积注意力建立数学联系，不意味着经典二值网络等同于完整 Transformer。
- Elman RNN 复用跨时间的更新函数；BPTT 展开训练时，长乘积可带来梯度消失或爆炸，LSTM 用门控维持记忆。时间递归每步更新观测；现代循环深度则在一次输出内迭代潜状态。
- 停止方式不同：固定轮数给出可预估的计算量；ACT 学习停止、加计算惩罚；DEQ 把固定点作为输出约束并用求根近似。它们不等价，普通循环模型也不一定收敛到固定点。
- Transformer 的注意力提供序列内直接交互；Universal Transformer 把共享层循环放回深度，可按位置动态退出。Ouro 循环整栈，Huginn 只循环中间核心并各轮注入编码输入。算术与多跳合成任务上的发现不可外推为同预算下普遍占优。
- 自回归离散 token、单次预测中的深度循环、跨位置连续隐状态是三条可并存的计算通路。Feedback Transformer 汇总历史层表示；Coconut 在潜思考步用隐状态取代文字；Full-bandwidth Transformer 将上一位置隐状态与已采样 token 嵌入融合，并仍输出可读 token。不能据此说普通自回归模型只保存 token：它还可以用注意力读取历史 KV。

## 用到的一手论文口径

- [McCulloch–Pitts（1943）](https://doi.org/10.1007/BF02478259) · [Hopfield（1982）](https://doi.org/10.1073/pnas.79.8.2554) · [Hupé 等（1998）](https://pubmed.ncbi.nlm.nih.gov/9723617/)。
- [ACT（2016）](https://arxiv.org/abs/1603.08983) · [Universal Transformers（2018 预印本；ICLR 2019）](https://arxiv.org/abs/1807.03819) · [DEQ（2019）](https://arxiv.org/abs/1909.01377)。
- [Coconut（2024 预印本）](https://arxiv.org/abs/2412.06769) · [Huginn（2025）](https://arxiv.org/abs/2502.05171) · [Ouro（2025）](https://arxiv.org/abs/2510.25741) · [Full-bandwidth Transformer（2026）](https://arxiv.org/abs/2608.08888)。

## 对 wiki 的映射

- [通用循环模型：循环深度与一手证据](../../wiki/concepts/general-looped-models.md)（沿用既有概念节点；这里的历史叙事不作为论文或模型原文）。
