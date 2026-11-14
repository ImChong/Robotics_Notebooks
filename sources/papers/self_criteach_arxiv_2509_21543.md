# Self-CriTeach — 原文核对记录

- **标题：** Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation
- **作者：** Jinbang Huang、Zhiyuan Li、Yuanzhao Hu、Zhanguang Zhang、Mark Coates、Xingyue Quan、Yingxue Zhang
- **发表：** ICML 2026，PMLR 306:46825–46850
- **原文：** <https://arxiv.org/html/2509.21543>（v4，2026-06-22）；<https://proceedings.mlr.press/v306/huang26v.html>
- **PDF：** <https://arxiv.org/pdf/2509.21543>
- **项目：** <https://markli1hoshipu.github.io/Plan_LLM/>
- **源码：** <https://github.com/markli1hoshipu/Plan_LLM>
- **入库日期：** 2026-11-14
- **核对范围：** arXiv HTML v4 的方法 §3–5、实验 §6–9、局限 §10、附录 A 的数据生成和真实机器人实现要点，并与 PMLR / 官方 README 对照。
- **通读：** 已完成

## 原文要点

1. §5.1 从 LLM 生成并修复 PDDL domain，利用符号求解器合成问题与计划。
2. §5.2–5.3 将状态变化、可行性判断、回溯等符号求解过程转为 CoT，供 SFT 学习。
3. §5.4 将自生成的 domain 复用为目标谓词比例奖励与前置条件违反等约束成本，进行 CPO 后训练。
4. 表 3：SCT-4B 总体规划成功率 0.46，原始 Qwen3-4B 为 0.26；仅 SFT 0.43，单独 CPO 0.31（该表同一评测口径）。
5. 表 5：UR5e 实验包含 Room / Lab 两类任务；规则感知下 SCT-4B 为 0.80 / 0.90，对照符号求解器为 0.70 / 0.70。
6. §10：依赖足够强的基础模型和可符号化任务；连续接触、形变物体、复杂几何未充分解决。

## 开源边界

官方 `Plan_LLM` 仓含 SFT、CPO 与评测脚本及权重入口，但 README 明确称自动生成 planning domain 的组件尚待从另一个工作集成；**不能宣称完整闭环从零可复现**。

## 对 wiki 的映射

- [Self-CriTeach 论文实体](../../wiki/entities/paper-self-criteach.md)
