# Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization

> 来源归档（ingest）

- **标题：** Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization
- **类型：** paper / vla / vbts / force-control / tactile-reasoning / contact-rich-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2507.09160>
- **arXiv HTML：** <https://arxiv.org/html/2507.09160>
- **PDF：** <https://arxiv.org/pdf/2507.09160>
- **项目页：** <https://jialeihuang.github.io/tactileVLA.github.io/>
- **机构：** Tsinghua University；UESTC；Shanghai Jiao Tong University
- **提交：** 2025-07-12（arXiv v1）
- **入库日期：** 2026-09-23
- **一句话说明：** **视–触–语–动深度融合 VLA**：token 级 cross-attention 融合 + **混合位置–力控制器** + 可选 CoT 推理；UMI 夹爪双 VBTS + GoPro 采集；USB/Charger 插入 **35%/90%** vs π₀-base **5%/40%**；力相关语言 **零样本** 泛化（softly **4.68 N** vs hard **9.13 N**）。

## 开源核查（2026-09-23）

| 项 | 状态 |
|----|------|
| 项目页 | <https://jialeihuang.github.io/tactileVLA.github.io/> — Paper 按钮；**Code & Data** 区块 |
| GitHub / 权重 | 项目页 **GitHub — Coming Soon**；**未列** 可运行仓库或 checkpoint URL |
| 数据 | 项目页未提供公开下载链接 |
| 结论 | **待发布**（截至核查日仅论文 + 演示视频） |

## 摘要级要点

- **问题：** VLA 语义强但接触丰富任务缺 **细粒度力控**；纯视觉难感知遮挡下接触状态。
- **核心发现：** VLM 先验已含 **物理交互语义**；少量 demo 接触觉即可 **激活** 力相关理解与零样本泛化。
- **架构：** 预训练 VLM 编码 RGB + 语言 + 触觉 + 本体；**tactile-aware action expert** 输出目标位置与力；混合位置–力控制执行。
- **能力轴：** (1) 触觉感知指令跟随；(2) 触觉常识抓取；(3) 触觉参与自适应推理（Tactile-VLA-CoT）。
- **擦板任务：** 域内 **80%**；域外黑板 CoT **80%**（基线 OOD **0%**）。
- **与 Awesome Touch 索引：** [`sources/papers/sun_awesome_touch_2507_09160_tactile-vla-unlocking-vision-language-ac.md`](./sun_awesome_touch_2507_09160_tactile-vla-unlocking-vision-language-ac.md) 为清单级摘录；**全文归档以本文件为准**。

## 核心论文摘录（MVP）

### 1) Token 级多模态融合 + 混合位置–力控制

- **链接：** <https://arxiv.org/abs/2507.09160> §Method；项目页 Figure 1
- **摘录要点：** 视觉、语言、触觉、本体经 VLM 编码后 **自由 cross-attend**；action expert 同时预测 **位置与力目标**，由混合控制器执行。
- **对 wiki 的映射：**
  - [Tactile-VLA（Awesome Touch 实体）](../../wiki/entities/paper-sa-2507-09160-tactile-vla-unlocking-vision-language-action-mod.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) USB / Charger 插入与力语言零样本

- **链接：** 项目页 Quantitative Results
- **摘录要点：** Charger **90%** vs π₀-base **40%**；力词 softly/hardly 在未见 Charger 上仍分离 **4.68 N / 9.13 N**；基线力–语言无相关。
- **对 wiki 的映射：**
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)

### 3) Tactile-VLA-CoT 自适应推理

- **链接：** 项目页 Adaptive Reasoning
- **摘录要点：** 擦板力不足时 CoT 分析触觉反馈并增力；域外黑板 **80%** vs 基线 **0%**。
- **对 wiki 的映射：**
  - [T-Rex](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md) — 同轴「触觉 + 推理/反应」VLA 族对照

## 对 wiki 的映射（汇总）

- 实体页：[Tactile-VLA（arXiv:2507.09160）](../../wiki/entities/paper-sa-2507-09160-tactile-vla-unlocking-vision-language-action-mod.md)
- 方法/概念：[VLA](../../wiki/methods/vla.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
- 项目页归档：[`sources/sites/tactile-vla-jialeihuang.md`](../sites/tactile-vla-jialeihuang.md)

## 当前提炼状态

- [x] 架构、三能力轴、定量结果、项目页 Coming Soon 开源核查已摘录
- [x] 与 [`sources/sites/tactile-vla-jialeihuang.md`](../sites/tactile-vla-jialeihuang.md) 互证

## BibTeX

```bibtex
@article{huang2025tactilevla,
  title={Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization},
  author={Huang, Jialei and Wang, Shuo and Lin, Fanqi and Hu, Yihang and Wen, Chuan and Gao, Yang},
  journal={arXiv preprint arXiv:2507.09160},
  year={2025},
  url={https://arxiv.org/abs/2507.09160},
}
```
