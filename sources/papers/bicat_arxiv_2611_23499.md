# BiCAT｜Bimanual Coordination Action Tokenization

- **类型：** paper / arXiv:2611.23499
- **原文：** https://arxiv.org/html/2611.23499
- **摘要：** https://arxiv.org/abs/2611.23499
- **项目页：** https://magiclab-nus.github.io/BiCAT/
- **代码：** https://github.com/MAGICLAB-NUS/BiCAT
- **核查日期：** 2026-11-20
- **核对范围：** arXiv HTML §3 canonicalization/FSQ/tokenizer，§4 结果、消融与真机数据，部分实现附录；未逐段核完全部附录和引用
- **通读：** 未完成

## 原文记录

1. **方法：** 两臂先按符号映射与同一范围归一化到 canonical 动作域；共享编码器处理两臂块，通过对称 cross-arm attention 把伙伴状态带进 token；FSQ 量化、联合 decoder 重构。
2. **策略：** 半自回归两块解码：先并行预测左臂 token，再在其条件下并行预测右臂 token；并非“所有左右 token 完全同时”。
3. **评测：** RoboTwin 2.0 10 任务平均 69.3%，最强 8 基线之一 NAC 51.7%，优势 +17.6 个百分点；6 真机任务共 150 trial 成功 113，OAT 74、FAST 25。
4. **计算：** H200 单 action chunk 40.5 ms（token 生成测量），不可当总闭环时延；FSQ 1920 code 优于 4375 code 方案，token 数非越大越好。
5. **边界：** 6 个真机任务仅覆盖特定机型与布置；Handover Mic 等并非每项都领先；必须区分重构指标和下游成功率。

**核查状态：** 已核对核心方法与结果，但未逐页读完完整论文附录。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-bicat.md)
