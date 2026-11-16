# 被动步行器柔顺踝与信息复杂度度量（2611.13779）

> 论文一手原文归档

- **英文题目：** Softness is Something You also Need! Why Compliance Helps Control: Information Metrics of a Passive Walker with Compliant Ankle
- **arXiv：** https://arxiv.org/abs/2611.13779
- **HTML 全文：** https://arxiv.org/html/2611.13779v1
- **PDF：** https://arxiv.org/pdf/2611.13779v1
- **机构：** Fabio Bonsignorio 等；论文列有萨格勒布大学与巴西利卡塔大学等作者单位
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 正文 I–VIII 简化被动行走器、可行相空间/核复杂度、讨论和 Supplemental I–IV 的蒙特卡罗/核体积估计；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** 在最简被动步行器、带膝踝柔顺、足底滚动与质量分布模型上，考虑机械能、碰撞与足跟冲击重置，构造可持续步行的相空间。用 Monte Carlo 采样估计可行体积，并以 RKHS 核协方差与 logdet 分析分布复杂度；将几何支持集和动力学诱导概率测度分开。

## 原文摘录（提炼）
1. **问题与动机：** 机械柔顺性除了改变物理可行域大小，也可能改变系统实际经过的状态分布，从而影响控制负担。
2. **方法：** 在最简被动步行器、带膝踝柔顺、足底滚动与质量分布模型上，考虑机械能、碰撞与足跟冲击重置，构造可持续步行的相空间。用 Monte Carlo 采样估计可行体积，并以 RKHS 核协方差与 logdet 分析分布复杂度；将几何支持集和动力学诱导概率测度分开。
3. **实验：** 以理论推导和数值示意为主；附录使用 MC 接受比例估算可行状态体积，以 Gram 矩阵 regularized logdet 作有限样本 proxy，明确不等同算子级 RKHS 复杂度。原文没有真机策略训练或主动控制器计算量比较。
4. **边界与局限：** 模型是简化被动动力学而非完整人形/四足；数值受核参数、采样和足跟撞击假设影响。不可把「结构柔顺」等价于已经证实的真机低算力控制。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [Softness and Compliance](../../wiki/entities/paper-softness-compliant-ankle.md)
