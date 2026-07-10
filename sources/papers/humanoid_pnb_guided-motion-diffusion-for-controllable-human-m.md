# Guided Motion Diffusion for Controllable Human Motion Synthesis

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Guided Motion Diffusion for Controllable Human Motion Synthesis
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2305.12577>
- **入库日期：** 2026-07-10
- **一句话说明：** 去噪扩散在文本条件的人体动作合成上很有前景，但纳入空间约束（如预定运动轨迹与障碍物）仍难——而这对连接孤立动作与其周遭环境至关重要。GMD（Guided Motion Diffusion）把空间约束注入动作生成：① 提出有效的特征投影方案，操纵动作表示以增强空间信息与局部姿态的一致性；② 配一个新的插补公式（imputation formulation），使生成动作可靠遵循全局运动轨迹等空间约束；③ 针对稀疏空间约束（如稀疏关键帧）易在反向步骤中被忽略的问题，提出稠密引导（dense guidance），把稀疏信号转成更密的信号去引导生成。实验证明 GMD 在文本动作生成上显著超 SOTA，同时支持轨迹跟随与避障等空间控制。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-guided-motion-diffusion-for-controllable-human-m](../../wiki/entities/paper-notebook-guided-motion-diffusion-for-controllable-human-m.md).

## 对 wiki 的映射

- [paper-notebook-guided-motion-diffusion-for-controllable-human-m](../../wiki/entities/paper-notebook-guided-motion-diffusion-for-controllable-human-m.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis.html>
- 论文：<https://arxiv.org/abs/2305.12577>
