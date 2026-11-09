# Terrain-Indexed Motion Priors and Geometric Support for Extreme Humanoid Parkour（arXiv:2611.08056）

> 来源归档（paper）

- **标题：** Terrain-Indexed Motion Priors and Geometric Support for Extreme Humanoid Parkour
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08056>（v1，2026-11-05，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.08056v1>
- **作者：** Chenghao Sun、Youhao Lyu（共同一作）、Zizheng Li、Shuting Zhang、Ziyang Dong、Wenhao Cui（通讯）
- **机构（以论文首页脚注为准）：** 松延动力 Noetix Robotics（北京；Sun、Dong、Cui）；中国科学技术大学 USTC（Lyu）；国防科技大学 NUDT（Zhang）；Zizheng Li 的机构字段在 v1 中仍是占位符「[Institution Name], [City], [Country]」
- **代码：** 论文正文与 arXiv 摘要页均未给出代码 / 项目页链接（截至 2026-11-09 入库日）
- **入库日期：** 2026-11-09
- **一句话说明：** 三阶段渐进模仿：特权模仿准备可执行参考，地形索引 AMP 习得技能，再降低 AMP 权重并加入手 / 膝几何支撑引导；Noetix E1 上仿真混合赛道成功率 96.34%，真机零样本 29/30 单障碍、4/5 混合赛道。
- **核对范围：** 下载 arXiv v1 PDF 全文（8 页，含摘要、引言、相关工作、方法 III-A 至 III-C、实验 IV-A 至 IV-D、结论、参考文献），用 pdftotext 逐段读完并对照 Fig. 2 页面渲染图；公式 (1)-(13)、表 I 至表 V 均核对。论文无附录或补充材料页（PDF 仅 8 页），论文所称视频 / 补充材料未提供，故未读；未运行任何代码，未读被引论文原文。
- **通读：** 已完成

## 开源状态

- 论文与 arXiv 摘要页均无 GitHub / 项目页链接；按步骤 2.5 归为「未列链接」，入库日 2026-11-09 未发现代码，后续 lint 可跟进。

## 核心摘录

1. **三阶段设计（Sec. III-A）：** Stage 1 参考准备（privileged mimic 产出带力矩、接触、高度扫描的机器人参考 rollout）20,000 iter；Stage 2 地形索引 AMP 技能习得 20,000 iter；Stage 3 支撑适应 10,000 iter（共 50,000，其中 30,000 训练遍历策略，Sec. IV 训练细节与表 II）。
2. **地形索引 AMP（Sec. III-B）：** 共享运动特征主干 + 4 个地形判别头（High 0.95–1.05 m / Mid 0.55–0.65 m / Low 0.35–0.45 m / Flat，表 I）；判别器输入每帧 402 维（含 24 维接触、23 维力矩、231 维高度扫描），4 帧窗口；训练期用真实地形标签路由风格奖励，部署时无地形 / 技能标签。
3. **几何手 / 膝支撑（Sec. III-C）：** 训练期表面候选图（0.025 m 采样），每肢最多保留 8 个可达候选，预接触吸引 (Eq. 8) 与实际接触评估；支撑权重 α(n)=0.1+0.9·clip(n/96000,0,1)（Eq. 10），AMP 权重 β 由 1 降到 0.5。
4. **仿真结果（表 III / IV，每条件 2,048 episode）：** Ours Low/Mid/High/Hybrid/OOD = 100 / 98.44 / 98.14 / 96.34 / 93.46%；w/o Stage 2 混合 0%；w/o Stage 3 High 79.69%、混合 83.59%；支撑适应消融在 1 m×1 m 顶面平台上 Ours 93.46% 对 w/o Candidates 46.24%、w/o Relaxation 58.89%、w/o Stage 3 50.63%。
5. **真机（Sec. IV-D，表 V）：** Noetix E1（23 自由度，Jetson Orin NX 板载，深度相机光轴俯仰 60°），零样本部署，Low 10/10、Mid 9/10、High 10/10、混合 4/5、OOD 13/15；论文自述 Mid 失败为手臂支撑丢失，混合失败为突然转向后接 Mid 型翻转动作。

## 对 wiki 的映射

- [paper-terrain-indexed-parkour](../../wiki/entities/paper-terrain-indexed-parkour.md)
