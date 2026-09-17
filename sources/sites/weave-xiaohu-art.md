# WEAVE 项目页

> 来源归档（site）

- **标题：** WEAVE: Learning Whole-Body Dexterous Loco-Manipulation from Human–Object Interactions
- **类型：** site
- **链接：** <https://xiaohu-art.github.io/Weave/>
- **arXiv：** <https://arxiv.org/abs/2609.16683>
- **机构：** 清华大学；大连理工大学；香港中文大学
- **入库日期：** 2026-09-17
- **一句话说明：** 人–物 HOI → 可执行 robot–object reference → 统一 contact-aware PPO；G1+Inspire 九物体 dexterous loco-manipulation。
- **沉淀到 wiki：** [`wiki/entities/paper-weave.md`](../../wiki/entities/paper-weave.md)

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-17）。
- **[code](https://github.com/xiaohu-art/Weave)**、**[datasets](https://huggingface.co/datasets/appolyn/Weave/tree/main)**、**[pdf](./Weave.pdf)** 均已链出；页内 arXiv 按钮仍为注释占位，BibTeX 指向 **2609.16683**。

## 项目页要点

- **四阶段可视化：** SMPL-X → Whole-body retarget → Kimodo completion → Policy rollout。
- **Retarget：** 骨盆水平约束 + 地面高度；IK 后 **contact-aware hand refinement**（指尖贴面、力闭合、防穿透）。
- **Kimodo：** 补全「走到物体旁」locomotion 前缀；多方向采样扩 reference。
- **Policy：** 本体 + 物体 pose/几何 + 短 horizon reference → body + 近端指关节位置目标；奖励含 robot/object tracking、hand opposition、contact matching。
- **物体：** Tripod、椅、衣架、桌、灯、箱等 9 类；train/test split 见项目页表格。
- **Release：** ~23 h 物理 rollout + 接触标注，供下游 HOI 策略与物理一致 motion 生成。
