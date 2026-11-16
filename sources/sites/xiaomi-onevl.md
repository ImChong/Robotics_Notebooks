# Xiaomi OneVL 官方项目主页

- **类型：** site
- **机构：** 小米具身智能研究团队（Xiaomi Embodied Intelligence）
- **官网：** <https://xiaomi-embodied-intelligence.github.io/OneVL/>
- **论文：** <https://arxiv.org/abs/2604.18486>（NeurIPS 2026）
- **代码：** <https://github.com/xiaomi-research/onevl>
- **训练代码：** <https://github.com/GeorgeLuImmortal/OneVL_training>
- **模型权重：** <https://huggingface.co/collections/xiaomi-research/onevl-models>
- **入库日期：** 2026-11-09

## 项目主页核对

项目主页展示 OneVL 的框架图、视觉/文本潜推理解释、NAVSIM/ROADWork/Impromptu/APR1 指标，以及论文、代码与权重入口。论文页与仓库交叉指向同一 OneVL 项目，不拆成两个 wiki 节点。

**开放状态：** 推理代码、训练代码、模型权重已提供公开入口；完整复现实验仍依赖数据集获取、视觉 tokenizer 权重和环境。公开代码不代表第三方已重现所有论文结果。

## 机制与指标核对

- 双辅助解码器在训练期间约束潜空间：语言重构 CoT，视觉预测未来 0.5/1.0 秒的视觉 token。
- 最快主规划通路为预填潜标记、仅自回归预测轨迹，辅助解释为可选能力。
- NAVSIM 论文报告 88.84 PDM-score / 4.46 s；显式 CoT 88.29 / 6.58 s。

## 对 wiki 的映射

- [OneVL 方法与评测](../../wiki/entities/paper-xiaomi-onevl.md)
- [论文全文归档](../papers/onevl_arxiv_2604_18486.md)
- [推理与训练实现](../repos/xiaomi-onevl.md)
