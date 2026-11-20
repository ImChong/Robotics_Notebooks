# BiCAT｜Bimanual Coordination Action Tokenization

- **类型：** paper / arXiv:2611.23499
- **原文：** https://arxiv.org/html/2611.23499
- **摘要：** https://arxiv.org/abs/2611.23499
- **项目页：** https://magiclab-nus.github.io/BiCAT/
- **代码：** https://github.com/MAGICLAB-NUS/BiCAT
- **核查日期：** 2026-11-20
- **核对范围：** arXiv:2611.23499 HTML 正文 §1–5、参考文献，以及附录 §6–11（RoboTwin 任务/协议、训练数据与优化、基线 tokenizer、消融明细、左右臂表示、FSQ、三机型真机）与官方代码 README
- **通读：** 已完成

## 原文记录

1. **方法：** 两臂先按符号映射与同一范围归一化到 canonical 动作域；共享编码器处理两臂块，通过对称 cross-arm attention 把伙伴状态带进 token；FSQ 量化、联合 decoder 重构。
2. **策略：** 半自回归两块解码：先并行预测左臂 token，再在其条件下并行预测右臂 token；并非“所有左右 token 完全同时”。
3. **评测：** RoboTwin 2.0 10 任务平均 69.3%，最强 8 基线之一 NAC 51.7%，优势 +17.6 个百分点；6 真机任务共 150 trial 成功 113，OAT 74、FAST 25。
4. **计算：** H200 单 action chunk 40.5 ms（token 生成测量），不可当总闭环时延；FSQ 1920 code 优于 4375 code 方案，token 数非越大越好。
5. **边界：** 6 个真机任务仅覆盖特定机型与布置；Handover Mic 等并非每项都领先；必须区分重构指标和下游成功率。

**附录核查补充：** §6 RoboTwin 10 个任务每项 100 次 rollout，32 步 chunk 只执行前 16 步再重规划；训练/验证样本 2,375 / 125 个 episode。§7 中 14 维 RoboTwin 动作、PaliGemma2-3B 与 tokenizer 300k/policy 30k steps 的统一对照很关键。§8 表 9 / §9：cross-arm attention 与联合 decoder 同时保留才得到 69.3%；左右臂表征的 MMD 仅是特征对齐指标，不等于协调成功率。§10 表 10：词表 1,920 (69.3%)，大到 4,375 反降至 58.9%。§11 真机包括 Bimanual SO-101、G1、YAM；每任务 25 次测评，推理在 RTX 5090 上、机器人等待推理完成；真机端 32 步 chunk 全执行，与仿真 16 步策略不同。论文局限说明方法仍基于 joint-space、每个本体分别训练 tokenizer，跨本体和末端 SE(3) 适配尚待验证。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-bicat.md)
