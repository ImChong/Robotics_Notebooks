# Learning Object Manipulation from Scratch via Contrastive Interaction（arXiv:2606.11525）

> 来源归档（ingest）

- **标题：** Learning Object Manipulation from Scratch via Contrastive Interaction
- **短名：** IWR（Interaction-Weighted Resampling）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.11525>（v1 2026-06-10，cs.RO / cs.LG）
- **PDF：** <https://arxiv.org/pdf/2606.11525>
- **作者：** Tongle Shen、Caleb Chuck、Fan Feng、Biwei Huang
- **机构：** 加州大学圣地亚哥分校（UCSD）、德州大学奥斯汀分校（UT Austin）（arXiv HTML 署名；Fan Feng 与 Biwei Huang 标 equal advising）
- **会议：** CoRL 2026（项目页标注）
- **项目页：** <https://iwr-arxiv.github.io/>
- **代码：** 截至 2026-10-10 未列出
- **入库日期：** 2026-10-10
- **博客归档：** [aether_geometry_of_contact.md](../blogs/aether_geometry_of_contact.md)
- **一句话说明：** 把操作动力学写成分段光滑马尔可夫过程，证明交互引起的模式切换让 CRL 能量函数难以表示与规划；IWR 在交互前、中、后阶段做加权重采样，仿真平均 +19.8%，真机目标条件 air hockey 成功率 25% → 60%。

## 开源状态（步骤 2.5，2026-10-10）

- **结论：未列代码**。项目页只有方法、分析与结果表，无 GitHub 链接；arXiv 摘要只给项目页。

## 核心摘录（面向 wiki 编译）

- CRL 在运动与简单控制上表现好，但在交互丰富的操作上吃力；作者认为根源是 **以物体为中心的交互**（接触、抓取）引起底层动力学模式切换。
- 项目页的形式化：分解式 MDP（FMDP）+ 高斯插值。Lemma 1：交互发生时下一表示点只能用仿射变换 \(A_1\psi_t+b_t\) 局部近似 → 分段非线性。Proposition 1：操作中动作只在交互时生效，否则物体运动是被动的 → 能量函数误差被传播。
- IWR：围绕交互前 / 中 / 后阶段重采样，鼓励表示保住决定未来可达性的模式边界。
- 实验环境：2D 动态控制（Box2D）、机器人操作（Meta-World）、机器人 air hockey（仿真、sim-to-real、真机）。
- 基线：PPO、SAC、SAC+HER、SAC+HINT、CRL、CRTR。数值表见博客归档。

**对 wiki 的映射：** [paper-geometry-of-contact](../../wiki/entities/paper-geometry-of-contact.md)
