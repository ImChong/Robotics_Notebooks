# JEPA-Anything: Learning Predictive Models across Different Worlds

> 来源归档（ingest 关联资料）

- **类型：** arXiv 论文
- **arXiv：** <https://arxiv.org/abs/2609.20800>
- **作者：** Taoyong Cui, Zhongyao Wang, Xinyue Xu, Weiyang Liu, Zhaochen Yu, Yuying Zhang, Qiang Gao, Mengyue Yang, Wanli Ouyang, Pheng Ann Heng, Yingcheng Wu, Zhenfei Yin, Ling Yang
- **提交日期：** 2026-09-17
- **项目页：** <https://phai-labs.com/news/jepa-anything/>
- **代码：** <https://github.com/Gen-Verse/JEPA-Anything>（Apache-2.0）
- **一句话说明：** 以正交预测因子分解（OPF）把潜在目标拆成互补因子、分别预测并重组，尝试在不同动力学世界复用预测式表征学习。
- **为何入库：** 为 JEPA 从单一观测域迁移到多类科学与动力学任务提供框架化样本，并可与后续 RSI-Anything 的预测选择机制关联阅读。

## 核心摘录

1. **正交预测因子分解（OPF）：** 将未来潜在目标分解为互补因子，通过独立路径学习，再重组为共享预测设计；论文提出其为跨世界建模的统一结构。
   - **对 wiki 的映射：** [JEPA-Anything 实体页](../../wiki/entities/paper-jepa-anything.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)。
2. **多域评测：** 报告覆盖视觉、生物、临床轨迹、控制、分子动力学、物理场和天气等七类领域；十个动力学任务中，作者报告相较匹配 JEPA baseline 的各项指标改善。
   - **对 wiki 的映射：** 实体页「评测与结果」；需把论文报告的跨任务结论与公开仓库可复现范围分开。
3. **定量与外部证据：** Interventional Pong 单次干预误差降低 34.8%；四类分子系统报告最低的一步及 100 步误差；临床任务预测超过 1000 个事件；生物干预结果在共培养、类器官、肿瘤片段和小鼠中得到支持；Kepler 缩放斜率报告为 -1.4991。
   - **对 wiki 的映射：** 实体页「评测与结果」及「局限与风险」；不同任务的指标与验证材料不能合并成同一种证据等级。
4. **代码边界：** GitHub 仓库提供 typed PyTorch OPF 核心、任务设计验证/脚手架和合成结构示例；示例只检查数据及接口一致性，不训练模型、不产生论文基准结果，仓库不含领域数据集或训练权重。
   - **对 wiki 的映射：** 实体页「源码运行时序图」与开源边界说明。PhAI Labs 指向的独立 HF collection 不等同于 GitHub 仓库内含权重。

## 对应代码与官网

- [官方仓库归档](../repos/jepa-anything.md)
- [官方项目页归档](../sites/jepa-anything.md)
