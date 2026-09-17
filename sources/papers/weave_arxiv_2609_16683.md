# WEAVE（arXiv:2609.16683）

> 来源归档（paper）

- **标题：** WEAVE: Learning Whole-Body Dexterous Loco-Manipulation from Human–Object Interactions
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.16683>
- **PDF：** <https://arxiv.org/pdf/2609.16683>（项目页镜像 [`Weave.pdf`](https://xiaohu-art.github.io/Weave/Weave.pdf)）
- **项目页：** <https://xiaohu-art.github.io/Weave/>
- **机构：** 清华大学（Tsinghua University，IIIS / College AI）；大连理工大学（DUT）；香港中文大学（CUHK）
- **作者：** Liu Cao, Xingze Wu, Jingzhi Cui, Botian Xu, Mingzhi Pei, Ruoqu Chen, Mengdi Xu
- **入库日期：** 2026-09-17
- **一句话说明：** 从 SMPL-X 人–物 HOI 经接触感知重定向 + Kimodo 接近段补全 → 单策略 PPO 联合跟踪 G1+Inspire 全身与物体；九物体 92.5% 训练序列成功、未见序列 65.0%。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-17）：[`xiaohu-art/Weave`](https://github.com/xiaohu-art/Weave) 含 `install.sh`、`scripts/rsl_rl/train.py`、Hydra 配置；[`appolyn/Weave`](https://huggingface.co/datasets/appolyn/Weave) 发布 ~23 h 物理 rollout 与接触标注。

## 核心摘录

1. **问题：** 人形–物体交互需同时协调平衡、locomotion 与灵巧手接触力；人类示范可展示协调行为，但跨 embodiment/动力学迁移需学「如何建立并维持有效接触」。
2. **管线：** SMPL-X 人–物序列 → **Whole-body IK 重定向** + **接触感知 hand refinement**（力闭合 + 防穿透）→ **Kimodo** 合成接近 locomotion 前缀 → **contact- & geometry-aware PPO** 跟踪 reference。
3. **平台：** Unitree **G1**（29 body DoF）+ 双 **Inspire** 手（各 6 主动指关节，共 12 finger DoF）。
4. **数据：** 9 日常物体；训练 **7,869** 条（**19.56 h**）、测试 **1,605** 条（**3.67 h**）reference；另释 **~23 h** 物理执行 rollout（接触标注）。
5. **主结果（100k iter）：** 训练序列 **92.45%** success / **96.30%** progress；**未见 interaction 序列**（同物体）**64.98%** / **84.52%**。
6. **多物体联合 vs 专精：** 27k iter 联合策略测试 **95.26%**，九专精聚合 **91.48%**；专精在 7/8 跟踪误差更低，但完成率未必更高。
7. **架构/优化：** SimBaV2 + **Muon** 在小桌任务 3k iter 样本效率优于 MLP+AdamW。

**对 wiki 的映射**

- [paper-weave](../../wiki/entities/paper-weave.md)
