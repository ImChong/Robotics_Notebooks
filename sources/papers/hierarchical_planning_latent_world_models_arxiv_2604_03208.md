# Hierarchical Planning with Latent World Models（arXiv:2604.03208）

> 来源归档（paper / HWM）

- **标题：** Hierarchical Planning with Latent World Models
- **arXiv：** <https://arxiv.org/abs/2604.03208>
- **PDF：** <https://arxiv.org/pdf/2604.03208v2>
- **HTML：** <https://arxiv.org/html/2604.03208v2>
- **项目页：** <https://kevinghst.github.io/HWM/>
- **代码：** https://github.com/kevinghst/HWM_PLDM
- **作者：** Wancong Zhang、Basile Terver、Artem Zholus、Soham Chitnis、Harsh Sutaria、Mido Assran、Randall Balestriero、Amir Bar、Adrien Bardes、Yann LeCun、Nicolas Ballas
- **机构：** FAIR at Meta；纽约大学；Mila；布朗大学（以论文首页署名为准）
- **版本：** v1 2026-04-03；v2 2026-06-16（本归档以 v2 为主，v1 差异见下）
- **许可：** arXiv 页面列 CC BY 4.0
- **入库日期：** 2026-10-06（2026-11-12 通读升级）
- **一句话说明：** 在共享 latent 空间训练多个时间尺度的 world model，高层以 latent macro-action 规划子目标，低层以原始动作追踪子目标，全程 zero-shot MPC。
- **核对范围：** 通读 arXiv v1（PDF 全文 26 页，含附录 A–J）与 v2（PDF 全文 29 页，含附录 A–J 与 Table 17 逐配置试验表）的 pdftotext 全文，逐段读正文、表格与图注；图像本身（Fig. 1/4/10/12 的 rollout 画面、Fig. 5–9 的曲线）只读图注与可提取的坐标文字，未逐图目视核对；未运行 HWM_PLDM 代码，未读项目页视频；引用文献仅读条目，未读被引论文。
- **通读：** 已完成

## 论文要点（作者自述，章节对应 v2）

1. **问题（§1）**：单层 latent world model MPC 在两种情形失败：非贪心任务（最优轨迹需暂时远离目标，目标匹配代价非单调，局部优化器陷入贪心捷径）与长时程任务（预测误差累积、搜索空间随 horizon 指数增长）。
2. **方法（§2）**：低层模型 P(1) 预测一步 latent；高层模型 P(2) 以 latent macro-action 预测下一个 waypoint；macro-action 由 transformer 动作编码器（CLS token 经 MLP）把 waypoint 之间的低层动作片段压缩而来，不假设固定高层步长。高层最小化 L1 目标距离 ‖z_g − P(2)(l;z_1)‖1，展开后的第一个预测 latent 作为低层子目标，低层再最小化到该子目标的 L1 距离，每 k 步重规划。两层共用同一编码器与 latent 空间，不需要逆模型、技能学习或目标条件策略（§1、§2.3）。
3. **真机（§3.1，Table 1）**：Franka + Robotiq，7-DoF，单张目标图像；世界模型在约 130 小时 DROID（96h）与 RoboSet（30h）无标注数据上训练（附录 B.2）。10 组 pick-and-place 起止对 × 5 次 = 每物体 50 次；抽屉 7 个任务 × 5 次 = 35 次（v2 §3.1、附录 G）。HWM：杯子 70%、盒子 60%、抽屉 70%；单层 VJEPA2-AC 为 0%、0%、30%。
4. **手动子目标对照（v2 §4.1，Table 4）**：给单层 VJEPA2-AC 人工标注的抓取点子目标图像，杯/盒都到 80%；论文据此论证单层失败是规划问题而非世界模型表征不足。
5. **仿真（§3.2、§3.3，Table 2、3）**：Push-T（DINO-WM）d=75 时 17% → 61%（d=25：84%→89%，d=50：55%→78%）；Diverse Maze（PLDM，10×10 网格含外墙，25 个训练布局、20 个留出布局）最难档 D∈[13,16]：44% → 83%。
6. **算力（Fig. 5、§3.2、§3.3）**：Push-T 同等成功率约 3× 更少每步规划时间；迷宫 4× 更少。摘要写「up to 3×」。
7. **分析（§4）**：Fig. 6 低层模型在 ≤1 s 预测更准，高层单步预测在 ≥1.5 s 更准；Table 5 迷宫上学习的 macro-action（D∈[9,12]/[13,16] 为 95/83）远高于直接拼接原始动作（52/37）；Table 6 Franka 离线对照中 macro-action 与专家动作的余弦相似度 0.88 ± 0.03 vs delta-pose 0.80 ± 0.02；Fig. 7 macro-action 维度 ≥4 才能产生有效计划，更高维度的子目标更难被低层执行，Franka 取 4 维。
8. **容量对照（v2 附录 D.2，Table 16）**：更大的单层模型（DINO-WM 98M）并不更好（H=10/15 为 35%/15%，对比 44M 的 55%/17%，层级 94M 为 78%/61%）。
9. **作者承认的局限（v2 §7、§3.1）**：所有方法成功率随任务时程下降；单一 latent 空间贯穿各层，更长时程可能需要分层专用抽象；子目标由随机或定步长 waypoint 监督，质量受限；低层规划器的反馈未进入高层优化；失败主要来自感知不精确（深度估计）或差一点的执行，暗示高层粗子目标丢失了精细引导所需细节。

## v1 与 v2 的差异（逐项核对所得）

- v1 Table 1 把「手动子目标」列与端到端列并排，写作「VJEPA2-AC 80%/80%（Subgoals）」；v2 把它挪到 §4.1 Table 4，并明说 Table 1 全部是「单张目标图像、端到端」。
- v1 正文称 HWM 「consistently outperforms strong VLA baselines」；v2 改为「competitive … 目标指定接口不同（HWM、Octo 用目标图像，π 系列用语言目标）」，并指出约 77× 的数据量差。
- v1 Table 3 的 PLDM 平坦基线在 D∈[9,12] 为 63%；v2 改为 84%（v2 Table 16 中同档 54k 平坦模型为 85%，与 Table 3 的 84% 差 1 个点，论文未解释）。PLDM（hierarchy）两版均为 95%。因此 v1 的「差距随时程拉大」在中档变成较小的 11 个点。
- v2 新增 Table 17（逐配置成功次数与 Clopper–Pearson 95% 置信区间）、容量对照 Table 16、stride 敏感性 Table 15，以及 Table 5（迷宫上拼接原始动作的对照）。
- v1 贡献条目写「planning cost 降低最多 4×」，摘要写「up to 3×」；v2 贡献条目改为「up to a 3×」，4× 仅保留在 §3.3 迷宫段。
- v1 Table 1 标题写「Franka dual-gripper」，v2 改为「Franka」；Fig. 4/12 图注仍写「dual-gripper Franka arm」。附录 J.1 描述的是单臂 Franka Panda + Robotiq 双指夹爪。

## 关键实验细节（附录）

- **规划器（附录 B/C）**：Franka 与 Push-T 用 CEM，迷宫用 MPPI。Franka 层级：高层 3000 样本、预测视野 2、动作 4 维；低层 800 样本、视野 2，k=1；单层基线 2400 样本、视野 6（v1、v2 附录表数值一致）。
- **低层世界模型**：Franka 沿用 V-JEPA 2-AC，冻结 ViT-g/16 编码器，约 300M 参数的 ViT 预测器，输入 256×256；高层同架构，waypoint N=3，片段 0.33–4 秒（附录 B.2）。Push-T 低层 DINO-WM（冻结 DINOv2，25M）、高层 75M（附录 B.3）。迷宫低层 PLDM（编码器约 33k、预测器约 20k 参数，VICReg 防塌缩）、高层沿用冻结编码器，stride 10、8 维 macro-action（附录 B.4）。
- **VLA 基线（附录 J.1）**：Octo-base（DROID 微调、目标图像）；π0-FAST-DROID 与 π0.5-DROID 用语言目标，不微调，最终采用提示词 “put the cup on the plate” 且把盘子当目标位置标记；π0.5 杯子 100 次、盒子 50 次试验；抽屉任务因 Octo 评测时把抽屉弄坏而未跑 π 系列。
- **离线规划评测（附录 I）**：用 10 个 pick-and-place 任务和手工标注的抓取点子目标，比较规划出的动作和专家动作（余弦、L1），论文称其与真机成功「strongly correlates」，但论文内未给出相关性数字。

## 对 wiki 的映射

- [HWM 论文实体页](../../wiki/entities/paper-hwm-latent-world-model-planning.md)
- [VideoDB JEPA 长文实体页](../../wiki/entities/article-videodb-jepa-world-models.md)
- 站内已有：[LeWorldModel](../../wiki/entities/paper-lewm.md)、[V-JEPA 2](../../wiki/entities/paper-vjepa2.md)
