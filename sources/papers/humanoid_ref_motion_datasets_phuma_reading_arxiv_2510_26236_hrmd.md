# PHUMA 论文通读（参考运动数据集选型视角）

> 来源归档（原文通读 · 支撑 `wiki/comparisons/humanoid-reference-motion-datasets.md`）

- **标题：** PHUMA: Physically Reliable Humanoid Locomotion Dataset
- **作者：** Kyungmin Lee, Sibeen Kim 等（KAIST / KRAFTON / Holiday Robotics）
- **类型：** paper / dataset / motion-retargeting
- **原文：** <https://arxiv.org/abs/2510.26236>（2026-11 通过 arXiv PDF v3 获取全文并用 pdftotext 通读）
- **一句话说明：** 在同一套 MaskedMimic / BeyondMimic 跟踪设置下横向比较 LaFAN1、AMASS、Humanoid-X、PHUMA，是对比页「规模 vs 物理可信 vs 是否预重定向」判断的一手实证。
- **核对范围：** 通读正文 §1–§6 与附录 A（预处理、数据集构成）、B（重定向对比、吞吐、SPIDER 初始化）、C（观测 / 奖励 / PPO 超参 / 域随机化）、D（自采视频评测、阈值、跨模拟器、GMR 参考、跌倒率）、E（来源与过滤消融、重定向质量影响、损失留一）、F（骨盆路径跟随）、G（H1-2 结果）；参考文献列表仅浏览，未逐条核对；未阅读 Humanoid-X、AMASS、GMR、BeyondMimic、MaskedMimic 等被引原文，也未核对 OMOMO / HiPHI / BONES-SEED 等本页其他数据集的原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **数据集对比（表 4，G1，成功率阈值 0.15 m）：** LaFAN1 2.4 h → 46.1%；AMASS 20.9 h → 76.2%（SINK 重定向）；Humanoid-X 231.4 h → 50.6%；PHUMA 73.0 h → 92.7%（PHUMA Test）。结论：规模与质量缺一不可，Humanoid-X 大而脏，LaFAN1 / AMASS 干净但覆盖不足。H1-2 上趋势一致（表 39）。
- **组成（附录 A.2，表 8）：** PHUMA 共 76.01K 片段 / 72.96 h，由 LocoMuJoCo、GRAB、EgoBody、LAFAN1（2.40 h）、AMASS（20.86 h）等 MoCap 与 HAA500、Motion-X Video（31.98 h）、HuMMan、AIST、IDEA400 及自采视频等组成；LaFAN1 与 LocoMuJoCo 以已重定向形式并入。
- **人类运动来源（§2.1）：** MoCap 精确但昂贵（LaFAN1 仅数小时、AMASS 偏行走）；视频重建可扩展但有抖动、脚滑、穿地。作者把 AMASS 描述为 walking-centric。
- **物理违规两类（§1）：** 视频转运动误估全局平移导致悬浮 / 穿地；重定向只顾关节对齐导致关节越限与脚滑。
- **策展（§3.1，附录 A.1）：** 30 Hz、零相位 4 阶 Butterworth（根平移 3 Hz，姿态 6 Hz）；88 个 SMPL-X 足部顶点 + 多数投票估计地平面（δ=2.5 cm）；切 4 s 片段；阈值：根 jerk < 50 m/s³、足接触分 > 0.6、骨盆高 0.6–1.5 m、骨盆 / spine1 到支撑域 < 6 / 11 cm；Humanoid-X 过滤后保留 27.1%。
- **PhySINK（§3.2）：** SINK + 关节可行性（98% 限位）+ 接地 + 防滑；G1 表 1：PhySINK 保真 94.8 / 可行 100 / 非悬浮 99.9 / 非穿地 96.8 / 非滑 89.7；IK 保真仅 27.6，GMR 非悬浮仅 14.7，SINK 非穿地 14.9。
- **策展消融（表 2）：** 全部过滤后 62 h，PHUMA Test 87.0 vs 未过滤 231 h 的 50.6；足接触过滤贡献最大（附录 E.2.3 留一）。
- **重定向影响（表 3）：** AMASS 上 PhySINK 79.5 > SINK 76.2 > GMR 61.7 > IK 52.8；附录 E.3 结论：运动保真是首要因素，保真到阈值后其他伪影收益递减。
- **阈值（附录 D.2）：** 0.5 m 阈值下各数据集差距小，0.15 m 才拉开；因此作者采用 0.15 m。
- **来源消融（附录 E.1）：** 仅视频子集优于仅 MoCap 子集（PHUMA Test 85.7 vs 75.2），作者归因于视频子集规模约两倍且覆盖更广；MoCap 基底加全部视频最优。
- **实机（表 5）：** BeyondMimic 扩展为多运动跟踪（未来轨迹经 motion encoder，附录 C.1.2），G1 零样本，PHUMA 策略 MPJPE 34.3 mm 比 AMASS 的 41.0 mm 低 16.3%；Humanoid-X 策略仿真跌倒率 18.6% / 13.1%，约为 AMASS（SINK）5.7% / 4.2% 的三倍，故未上实机（附录 D.5）。
- **骨盆路径跟随（附录 F，表 36）：** G1 上 PHUMA 84.5 vs AMASS 60.5（PHUMA Test），差距在动态动作（如跑步）上最大。
- **跨模拟器（附录 D.3，表 24）：** Isaac Gym → MuJoCo 零样本，PHUMA 75.0 / AMASS 62.1（PHUMA Test）。
- **局限（§5）：** 仅平地 locomotion，不含场景交互与不平地形；大量视频运动在策展中被丢弃；实机对比只有 AMASS；PhySINK 不能完全消除伪影。
- **代价（附录 B.2）：** PhySINK 约 0.92 FPS（RTX 3090），离线构建。

## 对 wiki 的映射

- [`wiki/comparisons/humanoid-reference-motion-datasets.md`](../../wiki/comparisons/humanoid-reference-motion-datasets.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2510.26236>
