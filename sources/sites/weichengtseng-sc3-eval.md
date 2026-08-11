# SC3-Eval Project Page

> 来源归档

- **标题：** SC3-Eval | Self-Consistent Video Generation
- **类型：** site / project page
- **URL：** <https://weichengtseng.github.io/sc3-eval/>
- **论文：** <https://arxiv.org/abs/2606.18610>
- **代码：** 项目页头部仅 **Paper** 按钮；无 Code / Hugging Face / 数据集入口（截至 **2026-08-11**）
- **静态页仓：** <https://github.com/WeiChengTseng/sc3-eval>（GitHub Pages 源码；**非**训练/推理实现）
- **机构：** 多伦多大学（University of Toronto）、矢量研究所（Vector Institute）、英伟达（NVIDIA）、物理智能（Physical Intelligence）、斯坦福大学（Stanford）、加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-08-11
- **一句话说明：** SC3-Eval 官方项目页：展示前向–逆向动力学一致性、跨视角一致性与测试时一致性三轴，以及 table bussing 真机策略评估相关图与消融视频。

## 开源核查（2026-08-11，步骤 2.5）

| 入口 | 状态 |
|------|------|
| Paper | 已挂链 — [arXiv:2606.18610](https://arxiv.org/abs/2606.18610) |
| Code / Weights / Dataset | **未挂链** — 页头无 Code 按钮；检索 [WeiChengTseng/sc3-eval](https://github.com/WeiChengTseng/sc3-eval) 仅为静态站点（`index.html` / `assets`），无可运行训练/推理入口 |
| 结论 | **确认未开源**（可复现训练栈与权重未发布；仅项目页与定性视频开放） |

## 页面内容要点

- **三一致性轴** — forward-inverse dynamics / multi-view / test-time consistency
- **Policy evaluation** — 七个实机 VLA 与真机成功率相关；offline open-loop vs online closed-loop；InD table bussing 与 OOD reverse bussing
- **Ablation** — cross-view inpainting、inverse dynamics joint training
- **Uncertainty** — 逆动力学不确定性作 early-termination 可视化

## 对 wiki 的映射

- 论文摘录：[`sources/papers/sc3_eval_arxiv_2606_18610.md`](../papers/sc3_eval_arxiv_2606_18610.md)
- 沉淀 **[`wiki/entities/paper-sc3-eval.md`](../../wiki/entities/paper-sc3-eval.md)**
