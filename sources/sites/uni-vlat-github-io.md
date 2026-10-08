# Uni-VLaT 项目页（uni-vlat.github.io）

> 来源归档（site）

- **标题：** Uni-VLaT — Whole-Body Tactile Adaptation of VLA Policies for Humanoid Loco-Manipulation
- **类型：** project-page
- **初始 URL：** <https://uni-vlat.github.io/>
- **arXiv v2 当前指向的项目页：** <https://ggkiller-air.github.io/Uni-VLaT/>
- **论文：** [arXiv:2609.35450](../papers/uni_vlat_arxiv_2609_35450.md)
- **机构：** 清华大学（Tsinghua University）、北京航空航天大学（Beihang University）、中国传媒大学（Communication University of China）、香港大学（The University of Hong Kong）
- **作者：** Zihao Wang、Shutong Liu、Siqi Zheng、Liu Cao、Ruoqu Chen、Rundong Liu、Yanchao Yang、Mengdi Xu
- **入库日期：** 2026-09-30；复核日期：2026-10-08
- **论文版本：** arXiv v2，2026-09-30
- **代码：** 当前页面标注 “Code coming soon”，未提供 GitHub / Hugging Face / ModelScope 仓库链接
- **一句话说明：** 全身触觉 + 触觉锚定多模态未来预测适配预训练 VLA；G1 五任务 75% 均值。

## 核查结论（步骤 2.5）

- **已公开：** 五任务 demo、方法说明、主结果、跨 VLA backbone、消融及作者机构
- **代码状态：** 页面仍写 “Code coming soon”，本次未发现可核实的官方训练/推理仓库链接
- **版本一致性：** arXiv 已于 2026-09-30 发布 v2，但官网首屏仍显示 “Paper coming soon / arXiv coming soon”；以 arXiv 页面作为论文版本与作者信息的权威记录
- **硬件：** Unitree G1 + Isaac-GR00T / π0.5 + SONIC 低层

## 页面要点摘录

- **触觉覆盖：** 胸、背、肩、上背、双臂共 **8 区域**；256 taxel/臂（Basket 分析用）
- **Back-Tap：** 触觉是主导因素；有触觉变体均可 **85–90%**
- **DP 基线：** 同数据 from-scratch Diffusion Policy 因 **安全约束** 部署前被拒，无 SR
- **局限：** 噪声与安装敏感；缺大规模触觉仿真；仅五任务代表性评测
- **官网内容边界：** 项目页摘要写明各任务约 50 demonstrations，主配置 20 次真机 rollouts；不能把展示视频或 ADC 曲线解读为额外统计基准
