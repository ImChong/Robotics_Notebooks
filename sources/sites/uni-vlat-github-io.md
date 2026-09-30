# Uni-VLaT 项目页（uni-vlat.github.io）

> 来源归档（site）

- **标题：** Uni-VLaT — Whole-Body Tactile Adaptation of VLA Policies for Humanoid Loco-Manipulation
- **类型：** project-page
- **URL：** <https://uni-vlat.github.io/>
- **论文：** [arXiv:2609.35450](../papers/uni_vlat_arxiv_2609_35450.md)
- **机构：** Tsinghua 等（页内 Anonymous Authors）
- **入库日期：** 2026-09-30
- **代码：** **截至 2026-09-30 未开源** — 全页无 GitHub / Hugging Face / ModelScope 链接
- **一句话说明：** 全身触觉 + 触觉锚定多模态未来预测适配预训练 VLA；G1 五任务 75% 均值。

## 核查结论（步骤 2.5）

- **已公开：** 五任务文字定义、方法两阶段图、主表、接触曲线、跨 π0.5/GR00T、预测上下文消融
- **未列链接：** Code / Resources 区无仓库；双盲页未挂机构 logo 外链
- **硬件：** Unitree G1 + Isaac-GR00T / π0.5 + SONIC 低层

## 页面要点摘录

- **触觉覆盖：** 胸、背、肩、上背、双臂共 **8 区域**；256 taxel/臂（Basket 分析用）
- **Back-Tap：** 触觉是主导因素；有触觉变体均可 **85–90%**
- **DP 基线：** 同数据 from-scratch Diffusion Policy 因 **安全约束** 部署前被拒，无 SR
- **局限：** 噪声与安装敏感；缺大规模触觉仿真；仅五任务代表性评测
