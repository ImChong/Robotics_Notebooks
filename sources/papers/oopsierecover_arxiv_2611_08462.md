# OopsieRecover: Learning Transferable VLA Corrections from Semantic Interventions / 从语义干预中学习可迁移的 VLA 纠错

- **类型：** 原始论文（arXiv: 2611.08462v1；2026-11-05）
- **论文：** https://arxiv.org/abs/2611.08462
- **HTML 全文：** https://arxiv.org/html/2611.08462v1
- **PDF：** https://arxiv.org/pdf/2611.08462
- **官方项目：** https://robin-lab.cs.utexas.edu/oopsierecover/
- **代码开放：** 项目页未找到正式公开可运行仓库（截至 2026-11-09）
- **核查日期：** 2026-11-09
- **核对范围：** 阅读 arXiv HTML v1 的 §I–VII（干预收集、门控代价生成/标定、候选动作选择、实验及局限）；核对官方项目页演示/资源链接；未执行实机实验。
- **通读：** 已完成

## 核心机制（源自论文正文）
1. 从少量人类语言/语义干预中获得失败原因描述，并定位涉及的目标物体或场景实体。
2. 将语义纠偏转为有激活条件的几何代价函数，利用干预前轨迹标定并检验代价，形成可复用的纠偏程序。
3. 测试阶段冻结 VLA：生成候选动作片段，以场景动力学预测未来状态并计算条件代价，挑选避免已知错误的候选。
4. 该工作主要是测试时 steering，而不是重新训练一个会全部动作的纠错 VLA。

## 原始实验数字及读法
- 作者报告真机 DROID 任务进度：从次优基线约 35.1% 提升至约 61.0%，相对提升约 74%；不是 74 个百分点。
- 论文分别报告少量语义纠偏示例与仿真、真机任务，应区分示例数、物体状态及比较基线。

## 局限及复现边界
- 代价表达依赖语言语义定位与场景几何正确；新失败类型未必被已有代价覆盖。
- 多候选预测和评分增加计算；冻结 VLA 的动作空间与预测器误差仍是瓶颈。

## 对 wiki 的映射
- [OopsieRecover 项目节点](../../wiki/entities/paper-oopsierecover.md)

## 参考一手资料
- [arXiv HTML v1](https://arxiv.org/html/2611.08462v1)
- [官方项目主页](https://robin-lab.cs.utexas.edu/oopsierecover/)
