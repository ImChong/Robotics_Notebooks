# GenTrack（arXiv:2608.01410）

> 来源归档（ingest）

- **标题：** GenTrack: Physical Alignment for Robot-Native Motion Generation and Zero-Shot Humanoid Tracking
- **短名：** GenTrack
- **类型：** paper / humanoid / motion-tracking / text-to-motion / flow-matching
- **arXiv：** <https://arxiv.org/abs/2608.01410>
- **PDF：** <https://arxiv.org/pdf/2608.01410>
- **HTML：** <https://arxiv.org/html/2608.01410>
- **会议：** AAAI 2027（用户指定；预印本 2026-08）
- **项目页：** 无
- **代码：** 论文未列仓库；检索未见官方 GitHub / HF → **确认未开源**
- **作者：** Zeyu Ling、Xinyao Yu、Renye Yan、Jikang Cheng、Zhanke Wang、Qing Shuai、Changqing Zou（通讯）
- **机构：** 浙江大学（ZJU）；北京大学（PKU）；腾讯（Tencent / Hunyuan）；之江实验室（Zhejiang Lab）。作者行编号：Ling / Yu / Zou = 1（ZJU），Yan / Cheng / Wang = 2（PKU），Shuai = 3（腾讯混元），Zou 兼 4（之江实验室）
- **版本：** arXiv:2608.01410（v1/v2，2026-08）
- **入库日期：** 2026-08-15
- **一句话说明：** 已有 robot-native 文本→运动生成器与全身 tracker 之间做在线互训：滞后 tracker 的闭环执行给生成器组相对奖励（FlowGRPO），新生成参考再扩 tracker 覆盖；不采集新数据。G1 上接 ProtoMotions 与 SONIC 两条骨干。

## 摘要级要点

- **问题：** 通才 tracker 的零样本覆盖绑在昂贵具身语料上。文本→运动能扩监督，但人体/重定向分布只是可执行分布的代理：运动学合理 ≠ 闭环可跟。单向管线要么冻结生成语料、要么冻结奖励 tracker。
- **方法：** GenTrack 从预训练生成器 \(G_{\theta_0}\) 与 tracker \(\pi_{\phi_0}\) 出发，交替：(1) 多样 robot-space 采样；(2) 上一轮冻结 tracker 闭环打分；(3) 组相对 FlowGRPO 对齐生成器；(4) 结构合法的生成参考与公开 AMASS/LAFAN 等量混合训 tracker。KL 锚到初始生成器 + 原文对监督 rehearsal + 真实参考 replay 防崩。
- **表示：** G1 每帧 **38D**（3D root 通道、6D 骨盆旋转、29 驱动关节）；平面位置/朝向 canonicalize，平面用位移、高度保持绝对。**GMD 重定向只离线一次**，在线环内不再跑 retargeter。
- **数据：** 生成器初始化/排练用内部语料 **357,472** 条 GMD→G1 文本–运动对；tracker 后训练用公开 **12,733 AMASS-G1 + 604 LAFAN1-G1** 与生成参考。评测冻结：LAFAN1-G1 / AMASS-test-G1 / Wild-G1-clean，以及私有 **1,024** prompt 生成套件。
- **开源（截至 2026-08-15）：** 无项目页、无 GitHub/HF；论文未承诺放代码。Wild-G1 与 1,024 生成测试集为私有。未建 `sources/repos/` / `sources/sites/`。
- **混名：** 勿与视觉多目标跟踪 *GenTrack*（arXiv:2510.24399）或 [SDU-VelKoTek/GenTrack](https://github.com/SDU-VelKoTek/GenTrack) 混淆。

## 核心摘录（面向 wiki 编译）

### 执行奖励（滞后 tracker，当前 trainee 权重为 0）

\[
R(\mathbf{q};\bar{\pi}^{(r)})=-S_{\mathrm{exec}},\quad
S_{\mathrm{exec}}=(1-c)+[e_j]_2+[e_t/0.5]_2+0.5[e_d/0.5]_2+2\mathbb{I}_{\mathrm{fall}}
\]

其中 \(c\) 完成率，\(e_j\) 最大关节误差（rad），\(e_t\) 根轨迹误差（m），\(e_d\) 根位移误差（m），\([x]_2=\min(x,2)\)。主协议不用速度/幅度/二值成功门控。

### Table 1：零样本 G1 跟踪（fall-only SR；误差含失败帧）

| 方法 | LAFAN1 | AMASS-test | Wild-G1 | MPJPE (mm) | \(E_g\) (mm) | MPJVE | RootVelErr |
|------|--------|------------|---------|------------|--------------|-------|------------|
| Any2Track | 100.0 | 5.1 | 10.4 | 320.9 | 1309.1 | 0.720 | 0.632 |
| BeyondMimic | 87.5 | 93.0 | 61.0 | 99.4 | 347.6 | 0.251 | 0.279 |
| Humanoid-GPT | 85.0 | 83.3 | 71.4 | 128.1 | 1134.7 | 0.689 | 0.644 |
| ProtoMotions \(T_0\) | 75.0 | 81.2 | 45.9 | 142.2 | 789.8 | 0.320 | 0.466 |
| SONIC | 85.0 | 79.0 | 47.2 | 126.2 | 814.2 | 0.308 | 0.423 |
| ProtoMotions GenTrack | 75.0 | 81.2 | 47.3 | 139.3 | 775.4 | 0.320 | 0.466 |
| **SONIC GenTrack** | **90.0** | **79.7** | **48.0** | **124.1** | **807.2** | 0.308 | 0.423 |

BeyondMimic 是 per-reference specialist，不是零样本通才对照。SONIC 支相对 \(G_0\) replay：三 split SR 全升，MPJPE −9.2 mm、\(E_g\) −40.4 mm。Final-\(G\) 离线 replay 复现不了在线轨迹。ProtoMotions 支 Wild-G1 最好、MPJPE 下降，但 \(E_g\) 不如冻结 \(G_0\) replay。

### Table 2：生成器（冻结 SONIC 执行 + TMR-G1）

| 生成器 | Succ.↑ | \(E_{\mathrm{joint}}\)↓ | \(E_{\mathrm{key}}\)↓ | TMR R@1 | FID↓ |
|--------|--------|-------------------------|-----------------------|---------|------|
| \(G_0\)（HYMotion 初始化） | 92.58 | 0.159 | 0.410 | 0.774 | 0.023 |
| Filtered SFT | **96.97** | 0.149 | 0.348 | 0.771 | 0.028 |
| Frozen tracker reward | 90.92 | 0.158 | 0.363 | 0.767 | 0.027 |
| GenTrack (ProtoMotions) | 93.55 | 0.160 | 0.399 | 0.782 | **0.020** |
| **GenTrack (SONIC)** | 94.43 | 0.152 | **0.325** | **0.783** | **0.020** |

Filtered SFT 把成功刷到 96.97，但是迎合冻结 \(T_0\) 的低难度偏置，语义/FID 变差。SONIC 支把 \(E_{\mathrm{key}}\) 从 0.410 降到 0.325，同时保住检索与 FID。

### Table 3：robot-native 源验证（附录）

同一冻结 SONIC 执行器上：Retargeted GT Succ. 91.41；KIMODO-G1 98.63 但 TMR R@1 仅 0.500；HYMotion \(G_0\) Succ. 91.89、R@1 0.774。说明初始生成器是 **HY-Motion 风格 robot-native 头**，不是 Kimodo 权重。

### 消融（ProtoMotions 初始化，Table 5–6）

- 只训 tracker / 只训生成器 / 去掉执行奖励 / 只用成功奖励 / **当前 trainee 当裁判**（跨 split SR 掉到 74.8/79.6/46.4）都复现不了全环。
- Reward-weighted SFT 原始成功最高（94.10）但 R@1/FID 掉到 0.769/0.029。
- 去掉 KL 或 GT rehearsal：执行成功更高，但检索/FID/多样性明显变差。

## 开源核查（步骤 2.5）

无独立项目页。论文、HTML、检索均未列 GitHub / Hugging Face / 项目站。评测用公开 ProtoMotions / SONIC checkpoint，但 **GenTrack 后训练代码、内部 357k 对、Wild-G1、1,024 生成套件均未发布**。→ **确认未开源**。未建 `sources/repos/` / `sources/sites/`。

## 对 wiki 的映射

- 升格 [GenTrack 论文实体](../../wiki/entities/paper-gentrack.md)
- 交叉：[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[ProtoMotions](../../wiki/entities/protomotions.md)、[HY-Motion 1.0](../../wiki/methods/hy-motion-1.md)、[PhyGile](../../wiki/entities/paper-phygile.md)、[PARC](../../wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md)、[RLPF](../../wiki/entities/paper-notebook-rl-from-physical-feedback-aligning-large-motion.md)、[人形运动跟踪选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[Kimodo](../../wiki/entities/kimodo.md)、[Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md)

## 当前提炼状态

- [x] 机制、两张主表、开源结论、混名警告
- [x] wiki 实体与交叉引用
