# PHUMA 论文通读（人形训练数据管线视角）

> 来源归档（原文通读 · 支撑 `wiki/queries/humanoid-training-data-pipeline.md`）

- **标题：** PHUMA: Physically Reliable Humanoid Locomotion Dataset
- **作者：** Kyungmin Lee, Sibeen Kim 等（KAIST / KRAFTON / Holiday Robotics）
- **类型：** paper / dataset / motion-retargeting
- **原文：** <https://arxiv.org/abs/2510.26236>（2026-11 通过 arXiv PDF v3 获取全文）
- **一句话说明：** 用「物理感知策展 + PhySINK 重定向」把 MoCap 与互联网视频合成 73 h 物理可信的人形 locomotion 参考库，是「来源 → 重定向 → 训练」三层管线的完整实证。
- **核对范围：** 通读正文 §1–§6 与附录 A–G（数据预处理、数据集构成、重定向对比与耗时、观测/奖励/PPO 超参/域随机化、评测细节、各类消融、H1-2 结果）；参考文献列表仅浏览，未逐条核对被引文献。未阅读 Humanoid-X、GMR、MaskedMimic、BeyondMimic 等被引原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§1）：** 视频转运动（Humanoid-X）有两类物理违规：视频到运动模型误估全局平移导致悬浮/穿地；重定向只顾关节对齐导致关节越限与脚滑。
- **管线（图 3）：** 策展 → PhySINK 重定向 → 策略学习 → 野外视频应用四阶段。
- **策展（§3.1，附录 A.1）：** 30 Hz 重采样，零相位 4 阶 Butterworth 低通（根平移 3 Hz，姿态 6 Hz）；用 SMPL-X 足部 88 个顶点对地面高度做多数投票（δ=2.5 cm）估计全局地平面；切成 4 s 片段，按根 jerk < 50 m/s³、足接触分 > 0.6、骨盆高 0.6–1.5 m、骨盆/脊柱到支撑域距离 < 6/11 cm 过滤；Humanoid-X 过滤后保留约 27.1%。
- **PhySINK（§3.2）：** 在 SINK（先把人体形状拟合到机器人再对齐姿态）上加关节可行性（98% 位置/速度限位）、接地、防滑三项损失。
- **重定向指标（表 1，G1）：** IK/GMR 因线性缩放丢运动保真；SINK 保真 94.8% 但穿地（非穿地 14.9%）、滑脚；PhySINK 五项指标均衡（保真 94.8、可行 100、非悬浮 99.9、非穿地 96.8、非滑 89.7）。
- **策展消融（表 2）：** 全部过滤后仅 62 h，PHUMA Test 成功率 87.0 对比未过滤 231 h 的 50.6；足接触过滤贡献最大。结论：数据质量重于数量。
- **重定向对训练的影响（表 3、附录 E.3）：** 在 AMASS 上 PhySINK 79.5 > SINK 76.2 > GMR 61.7 > IK 52.8；运动保真是影响跟踪的首要因素，保真到阈值后其他伪影收益递减。
- **数据集对比（表 4，G1，0.15 m 阈值）：** LaFAN1 2.4 h → 46.1；AMASS 20.9 h → 76.2；Humanoid-X 231.4 h → 50.6；PHUMA 73.0 h → 92.7。规模与质量缺一不可。
- **实机（表 5）：** BeyondMimic 扩展为多运动跟踪，G1 零样本，PHUMA 策略 MPJPE 34.3 mm 比 AMASS 的 41.0 mm 低 16.3%；Humanoid-X 策略仿真跌倒率约为 AMASS 的 3 倍，未上实机。
- **来源消融（附录 E.1）：** 仅视频子集优于仅 MoCap 子集，原因归于视频的规模与多样性；MoCap 基底上逐步加视频，全量视频最优。
- **局限（§5）：** 仅平地 locomotion，不含场景交互与不平地形；大量视频运动在策展中被丢弃；实机对比只有 AMASS；PhySINK 不能完全消除伪影。
- **代价（附录 B.2）：** PhySINK 约 0.92 FPS（RTX 3090），因整条轨迹联合优化，作者认为离线构建可接受。
- **其他：** 评测成功率阈值取 0.15 m（比常规 0.5 m 更能拉开差距）；MaskedMimic 用于仿真评测；PhySINK 作初始化可把 SPIDER 的开环成功率从 4.6% 提到 10.7%。

## 对 wiki 的映射

- [`wiki/queries/humanoid-training-data-pipeline.md`](../../wiki/queries/humanoid-training-data-pipeline.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2510.26236>
