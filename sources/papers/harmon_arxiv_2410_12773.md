# Harmon: Whole-Body Motion Generation of Humanoid Robots from Language Descriptions（arXiv:2410.12773）

> 来源归档（ingest · arXiv + 项目页）

- **标题：** Harmon: Whole-Body Motion Generation of Humanoid Robots from Language Descriptions
- **类型：** paper / humanoid / text-to-motion / vlm / loco-manipulation
- **arXiv：** <https://arxiv.org/abs/2410.12773> · PDF：<https://arxiv.org/pdf/2410.12773>
- **项目页：** <https://ut-austin-rpl.github.io/Harmon/> — 归档见 [`sources/sites/harmon-ut-austin-rpl.md`](../sites/harmon-ut-austin-rpl.md)
- **机构：** 德克萨斯大学奥斯汀分校（UT Austin）；英伟达研究院（NVIDIA Research）
- **入库日期：** 2026-09-27
- **一句话说明：** 用语言驱动 **Harmon**：PhysDiff 人体先验 + IK 重定向 + **VLM 常识编辑** 补头/指与修语义，上下身解耦后在 GR1 真机执行全身 motion。

## 开源状态（步骤 2.5）

- **核查日：** 2026-09-27。
- **已发布：** arXiv、项目页视频与 method 图。
- **未发布：** 官方代码仓库。
- **结论：** **未开源**；wiki `## 源码运行时序图` 标 **不适用**。

## 摘录 1：人体先验 + 重定向缺口

大规模 **language–motion 配对** 在人形侧稀缺，但人体动捕库丰富。Harmon 用 **PhysDiff**（带物理约束的扩散 text-to-human）产出 **SMPL** 序列，再 IK 映射到仿真人形。直接 retarget 的问题：（1）动捕缺 **头/指**；（2）运动学差异导致 **语义/可读性** 偏移。

**对 wiki 的映射：** 升格 [`paper-loco-manip-161-097-harmon`](../../wiki/entities/paper-loco-manip-161-097-harmon.md)；互链 [text-to-motion](../methods/diffusion-motion-generation.md)、[motion retargeting](../methods/motion-retargeting-gmr.md)。

## 摘录 2：VLM 运动编辑

给定 **渲染后的人形 motion** 与语言描述，VLM：（1）从原句抽取 **finger/head** 子描述并生成对应 motion；（2）**迭代** 评估渲染是否与语言一致并调整（尤其手臂）。页上 failure case 表明高频/大偏差/缺 primitive 时仍会失败。

**对 wiki 的映射：** 与「纯 retarget / 纯 T2M」对照；VLM 层是 **后处理编辑器** 而非端到端关节策略。

## 摘录 3：真机执行与上下身解耦

真机 **Fourier GR1** 两外观版本；locomotion 与 upper-body **分开控制** 以落地全身 motion。项目页含手语（GPT-4 分句）与自由文本跟做视频。

**对 wiki 的映射：** [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)；与 [ProtoMotions](../../wiki/entities/protomotions.md) / 跟踪栈（执行层）区分——Harmon 产出的是 **参考 motion 轨迹**，不是 RL tracker 本身。

## 摘录 4：与 MaskedMimic / ProtoMotions 谱系（读者对照）

Harmon 走 **语言→人体生成→VLM 修→人形**；NVIDIA **MaskedMimic**（TOG 2024，[PDF](https://research.nvidia.com/labs/par/maskedmimic/assets/SIGGRAPHAsia2024_MaskedMimic.pdf)）走 **物理角色 masked inpainting**，官方实现入口在 [**ProtoMotions**](https://github.com/NVlabs/ProtoMotions)。二者都服务「全身行为」，但 Harmon 强调 **VLM 语义编辑**，MaskedMimic 强调 **部分观测条件下的物理补全**。

**对 wiki 的映射：** [paper-bfm-17-maskedmimic](../../wiki/entities/paper-bfm-17-maskedmimic.md)、[protomotions](../../wiki/entities/protomotions.md)。

## BibTeX

```bibtex
@article{jiang2024harmon,
  title={Harmon: Whole-Body Motion Generation of Humanoid Robots from Language Descriptions},
  author={Jiang, Zhenyu and Xie, Yuqi and Li, Jinhan and Yuan, Ye and Zhu, Yifeng and Zhu, Yuke},
  journal={arXiv preprint arXiv:2410.12773},
  year={2024}
}
```
