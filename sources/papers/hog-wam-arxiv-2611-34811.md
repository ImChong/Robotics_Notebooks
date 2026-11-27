# HOG-WAM: Improving Visual Robustness through Future Gradient Prediction

- **类型：** 论文预印本
- **作者：** Boxuan Dong、Yupeng Qian、Ying Wu、Heshuai Li、Junhao Tan、Zhenxiong Tan、Weijia Wu、Weihao Yu
- **机构：** 北京大学、新加坡国立大学
- **来源日期：** 2026-11-25
- **原文：** [arXiv:2611.34811v1](https://arxiv.org/abs/2611.34811v1) · [HTML](https://arxiv.org/html/2611.34811v1) · [PDF](https://arxiv.org/pdf/2611.34811v1)
- **代码：** [iPPLab/HOG-WAM](https://github.com/iPPLab/HOG-WAM)（[源码归档](../repos/hog-wam.md)）
- **通读：** 已完成
- **核对范围：** arXiv v1 正文 §1–6、图 1–4、表 1–7；附录重点核对污染协议、HOG 构造、非对称注意力及去噪、Franka 试验与 Wilson 区间、替换消融。

## 技术摘录

1. 作者将确定性梯度栅格设为额外未来视觉目标，并以训练专用模块监督控制前的 RGB 表征。对 wiki 的映射：核心机制。
2. RoboTwin-Plus 固定任务和成功判据，只改观测；LIBERO Plus 传感器噪声提升而背景、照明表现下降。对 wiki 的映射：实验与评测。
3. 真实 Franka 的提升集中在堆叠等部分任务，只有一个噪声设定；噪声下插入两者均未完成。对 wiki 的映射：局限与风险。

对应 [HOG-WAM 节点](../../wiki/entities/paper-hog-wam.md)。
