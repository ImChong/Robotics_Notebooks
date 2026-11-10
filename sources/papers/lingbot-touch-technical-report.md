# LingBot-Touch — 官方技术报告全文通读（27 页）

> 一手论文精读归档。以下结果来自论文原文，而非模型卡、新闻转载或第三方复现。实验指标为作者报告，尚未在本地复现。

- **英文标题：** Learning Heterogeneous Tactile Representation for General Dexterous Manipulation
- **中文标题：** LingBot-Touch：面向通用灵巧操作的异构触觉表征学习
- **作者：** Bing Han, Yichuan Peng, Haoyu Dong, Yuanmeng Hu, Yongyi Jia, Fangjing Wang, Kecheng Zheng, Yiming Zhang, Yanhong Zeng, Xiaoqing Zhang, Chen Song, Rundong Zou, Anqi Dong, Yuan Guan, Ka Leong Cheng, Xing Zhu, Yujun Shen, Nan Xue, Wei Ning（† Project Lead）
- **机构：** Robbyant（蚂蚁灵波），与官方仓库和项目页一致；未为各作者猜测单位
- **原文 PDF：** <https://github.com/Robbyant/lingbot-touch/blob/main/paper/Learning_Heterogeneous_Tactile_Representation_for_General_Dexterous_Manipulation.pdf>
- **官方网站：** <https://technology.robbyant.com/lingbot-touch>
- **代码：** <https://github.com/Robbyant/lingbot-touch>
- **核验日期：** 2026-11-10
- **原文件：** Git LFS, 19,010,130 bytes，SHA-256 `1d68b6ce43470b9ffbdfd7425480581845b2515eb7897661e26152c5b257a3ce`。
- **读取方式：** 经临时 GitHub Actions 环境下载 Git LFS 实体，严格 SHA-256 校验后用 pdftotext -layout 按页提取；核对 PDF 27 页内容（含 1–19 页正文、20–24 页附录及 25–27 页参考文献）。未把原 PDF 或全文转存至知识库。
- **核对范围：** p1 abstract，§1 Introduction、§2 Related Work、§3.1–3.3 方法与式 (1)–(5)、§4.1–4.6 probing / scaling / objective / ablation、§5.1–5.4 真机与仿真 RL、§6 结论与局限；Appendix A（Tables 4–6）、B（Table 7 与 Figures 14–15）、C.1–C.3（Figure 16、Table 8），References [1]–[37]；核对 Figures 1–16 和 Tables 1–8 中的文字、数值及说明。
- **通读：** 已完成

## §1–`2：动机与相关工作

- 触觉传感器的物理原理、taxel 几何、采样频率和通道语义不统一；大部分点在大部分时间没有接触，简单重建易学习背景；单一传感器数据稀缺，不能简单迁移到陌生硬件。
- 对照：Sparsh、T3、AnyTouch/AnyTouch 2、FTP-1、HTT 等。LingBot-Touch 的核心区别是统一几何地址 + 三种模态边界 tokenizer + 共享 HaT + masked JEPA latent prediction；它是表征基座，不负责低层机器人控制。

## §3.1：数据组成（Figure 2；Appendix Table 4）

| 模态 | 来源 | 时长 | Episodes | 关键约束 |
|---|---|---:|---:|---|
| 压阻 | Juqiao / Xspark AI / TACHIN 人戴触觉手套 | 2,197 h | 86,906 | 9、16、或拇指 124 / 其余指 61 点；60–100 Hz |
| 磁性 | PaXini 人戴触觉手套 | 1,136 h | 294,871 | 拇指 116、其余指 96；约 41.7–83.3 Hz |
| 视触觉 | Sharpa Wave 双臂灵巧手遥操作 | 1,650 h | 82,654 | 原始变形图 240×240，30 Hz |
| **总计** | 24 类操作类别，真实人类/机器人接触 | **4,983 h** | **464,431** | 约 **1.33B frames** |

- 人戴手套占约 2/3、遥操作约 1/3。Episode 只有任务类别标签，不具有每个 taxel 的力/接触位置/时间语义标注。
- 三种模态在训练批次中等概率采样（不是按小时直接比例采样）。

## §3.2：Tactile-JEPA 与 Haptic Transformer（Figure 3；式 1–4）

1. **规范化空间：** 物理 taxel 几何先按传感器/手指用固定旋转、半径和 anchor 变换到 canonical fingertip sphere：`p_i = α p̃_i + b`（式 1）。每个 token 位置采用 5D 地址 `(t_i, f_i, x_i, y_i, z_i)`（式 2），5D RoPE 分别作用于注意力 Q/K；信号正向法向分量映到 [0,1]，带符号切向分量映到 [-1,1]。
2. **各模态 tokenizer：** magnetic、piezoresistive、visuotactile 各自两层 MLP；主模型 tokenization 把相邻 **2 帧同一 taxel 的 6 个信号数**映成一个 token；`T` 帧 `N` 个有效 taxel 构成 `L=N×T/2` tokens。共享 Haptic Transformer 学跨指、跨时、跨位置的上下文；小模型和大模型均 **12 层**。
3. **上下文 / 目标 / 预测器：** context encoder 只看未掩码 taxel，模态路由 predictor 将 visible representations 与 masked tokens 合并并预测全位置；EMA target encoder 处理全量输入，四个不同深度的 target 特征分别 LayerNorm 后拼接监督，target 路径 stop-gradient。
4. **损失（式 4）：** `L = mean_{i∈M}|ẑ_i - sg(z̄_i)|_1 + λ mean_{i∈C}|ẑ_i - sg(z̄_i)|_1`，两个均值均按拼接通道数归一，`λ=0.5`；masked target prediction 与 visible context alignment 同时存在。
5. **空间整点掩码：** mask 同一 taxel 全时间保持，防止模型从邻近帧偷看相同 taxel；最终空间掩码 **80%**，训练前 **300** optimizer steps 从 **40%→80%** 线性 warm-up，避免稀疏接触全部被遮蔽造成不稳定；论文称总计 **200,000** 次 optimizer updates。
6. **下游：** 使用**冻结 EMA target encoder/tokenizers**提取四层拼接特征，训练 task-conditioned attentive probe 或策略头；公开仓的推理包只有冻结 encoder，没有训练 predictor。

## §3.3：布局泛化（式 5）

- 同一训练 clip 随机挑多种采样密度：磁性每指 24/32/48/64/76/96 点；视触觉 32/96/200 点；低分辨率压阻保持原布局。
- 点位扰动：`p'_i = p_i + δ_i，||δ_i||₂ ≤ 0.5 d_i`，其中 `d_i` 为同指最近相邻有效点距离；每片段采样，同一 taxel 所有时序和 target/context 共用同一偏移，只改变位置编码，不改观测时序。
- **跨传感器零参数更新**指不微调触觉 encoder；新传感器仍需要可验证的轴向、定标、handedness 和 taxel 几何接口。

## §4.1–4.2：五项冻结 encoder 探针任务（Figure 5–6）

- 两平台：**Sharpa Wave 视触觉已见**，**XHand1 磁性传感器型号未见**；XHand1 120 点采样成 96。Probe 使用 cross-attention + 线性头、encoder 始终冻结，详细数据规模与 100-epoch、AdamW 训练预算见 Appendix Tables 5–6。
- 模型：Tiny **5.46M**（embed 192、12 layers），Small **21.76M**（embed 384、12 layers），此处默认使用 5K 数据预训练的 **Small**。
- 四个物理探针和一个触觉→视频检索：
  - Sharpa Wave：力 MAE **0.81 N**、中心误差 **0.12 mm**、接触占比 MAE **0.66 pp**、未来力 MAE **1.04 N**、retrieval top-5 **72.6%**；FTP-1 相应为 **2.31 N / 2.19 mm / 2.90 pp / 1.67 N / 16.5%**。
  - XHand1：**1.32 N / 0.16 mm / 0.43 pp / 1.45 N / 47.1%**；带下游传感器适配器的 FTP-1 为 **3.72 N / 5.89 mm / 4.39 pp / 2.00 N / 6.0%**。
- **公平性约束：** FTP-1 冻结 shared backbone，但**额外训练对应传感器 adapter + probe（100 epochs）**，Touch 只训练 probe；这些是本文定义的两种设置，不是所有技术路线的统一零样本排行榜。

## §4.3–4.6：scaling、反例、消融（Figures 7–10，Table 1）

- **容量/时长：** Small 从 1K→5K 时长在两个传感器的所有探针持续提升或持平；Tiny 在约 4K 饱和，5K 检索回落，**不能笼统宣称所有大小始终单调扩展**。
- **多源数据的两个不同比较：** 固定本传感器的数据量，再加入其他传感器时，8 组 sensor-task 中 **7 组改善**；但固定**总时长 1K**（单源全额 vs 三源混合），单源在自身传感器上的 **7/8 组**胜出，尤其空间精度。跨模态数据有迁移作用，不是固定预算下每项都占优。
- **JEPA vs VAE/MAE（Figure 9）：** 同数据与可比参数；1K 时 Touch center **0.90 mm** 对 VAE **3.03**、MAE **3.07**，retrieval recall **32.1%** 对 **4.4%/5.6%**；4K 时 Touch center **0.32 mm**、recall **49.3%**；signal-space reconstruction 在稀疏背景下可饱和或退化。跨不同损失数值**不可直接比较**收敛大小。
- **单因子消融（Table 1）：** 以 **1K / Tiny / 100 epochs** 的默认对照设置进行；默认设 **90% mask、4 帧、tubelet 1、固定 EMA=0.99925**，**并非最终最佳配置**。80% mask 相对 90% 在五项探针均更优；结合值和点位 augmentation、0.999→1.0 的 EMA schedule、按模态路由接口重要；context loss 的增益混合且较小。32 帧和 2/4 frame tubelet 在 30Hz 视触觉更适宜，不能无条件跨采样率沿用帧数。
- **可视化（§4.6 Figure 10）：** 16-frame window、stride 2、按五指 pooling 并拼接成 3,840 维再 UMAP；用阶段标注观察 noncontact 聚类与接触阶段连续结构。它是分析投影，不应把 pooling 当默认物理量读取策略。

## §5.1–5.2：26 真机任务、四策略（Table 2；Figure 11）

- Sharpa Wave **20** 项、XHand1 **6** 项。对照 **无触觉** / **原始触觉** / **冻结 Touch 表征**，四种策略 Diffusion Policy、π0.5、LingBot-VLA、T-Rex。每项 **20 次连续评测 trial**；每任务 **200** 条演示，需双臂的任务 **500** 条；成功判据有按子步骤计分规则，详见 Appendix Table 7。
- 触觉 encoder 为 **Tiny 5.46M，Table 2 注明预训练 4K 小时**（与部分 5K 数据 scaling 测试分开）；真实机器人使用 **30 Hz 的 16 帧触觉历史**、每手由可训练 cross-attention 聚合成 **40 tokens**，投影后作为原策略的附加条件输入。
- **四策略 26 任务均值（原始 → +Touch，Table 2）：**
  - Diffusion Policy **21.6% → 35.6%（+13.9 pp）**
  - π0.5 **48.1% → 69.9%（+21.8 pp）**
  - LingBot-VLA **48.3% → 73.1%（+24.8 pp）**
  - T-Rex **45.4% → 68.6%（+23.2 pp）**
- Touch 为 104 个 task×policy 组合中的 **89 组**最优、**99 组**达到或超过原策略；低于原策略 **5 组**、低于直接 raw tactile **9 组**。raw tactile 在 104 组中 **22 组反而降低**分数，不可宣称 tactile 永远带来增益。
- **XHand1 未见传感器单独切片均值**：LingBot-VLA **57.21%→75.04%**，而加 raw tactile 为 **57.21%**，这是跨硬件外推证据，不意味着 VLA 无需任务 post-training。
- **参数增加**：Touch 相对 T-Rex / π0.5 / LingBot-VLA 约 **0.16% / 0.15% / 0.13%**，不是完整端到端延迟占比。

## §5.3：仿真 PPO（Table 3、Figure 12；Appendix C）

- 两种手共 **12 个任务实例**（Sharpa 8、XHand1 4；归属 8 个任务类型），比较五种 observation：无触觉、binary contact、raw tactile、joint-trained tactile CNN、冻结 LingBot-Touch。
- **成功率跨任务等权均值**：**21.4% / 30.3% / 38.5% / 45.6% / 69.0%**。每项 **3 seeds**，Touch 成功率跨 seed 标准差最高 **9.2 pp**、平均 **3.9 pp**；PPO 训练曲线和 return 在 Table 3/Figure 12 分开呈现。
- Plate spinning（Sharpa）**97.4%** vs 最优其他 **82.9%**；pen spinning **97.1%** vs **64.2%**；并非每项任务都是相同的增量。
- Appendix C：仿真使用 TacMap 式 penetration-depth + ray query 在 Isaac Sim 中构建触觉变形图，8 帧、Sharpa 96 / XHand 120 点；冻结 encoder，PPO 策略头另行训练。额外 **7,200 s、47 objects** 仿真 probing；contact-positive Dice **0.708**、coverage MAE **1.089 pp**、force MAE **3.264 N**、future force-change MAE **0.459 N**，说明跨真实与模拟存在幅值误差。

## §5.4：表示质量与策略效果联系（Figure 13）

- 同一个 LingBot-VLA 主干、五项 Sharpa 真机任务，200 demos/task、20 trials/task。无触觉平均 **55.9%**；5K Tiny Touch **82.2%**；Touch 在 **1K/2K/3K/4K/5K** 预训练后分别为 **69.8/74.3/77.4/81.4/82.2%**；换 FTP-1 触觉 encoder 为 **67.5%**。
- 注意这里 **5K** 的实验属于图 13 单独的 scaling study，**不能**倒写 Table 2 使用 5K。

## §6 与附录边界

- **论文明确限制：** 真机与仿真手型仍需扩大；预训练 **未覆盖电容式**（作者现有数据未达质量要求）；主要涵盖指尖，手掌和其他手指表面接触覆盖不足。
- **Appendix A（Tables 4–6）：** 传感器/频率/样本量、五探针输入输出和训练预算；触觉→视频检索的视觉模型是 frozen V-JEPA 2 ViT-L，probe 用 symmetric InfoNCE，不应误以为触觉模型直接生成视频。
- **Appendix B（Table 7）：** 21 个任务名称在两种手上的覆盖组合共 26 个场景；部分任务按照子目标加分或序数评分，避免把分数一概理解成二值成功。
- **Appendix C（Table 8）：** 8 类仿真任务和各自成功终止判据；RL/真实操作均不等于从开源 checkpoint 一键重放。
- **参考文献 [1]–[37]：** 包括 I-JEPA、V-JEPA 系列、Sparsh、HTT、FTP-1、AnyTouch、TacMap、PPO 使用语境；不将本文性能宣称扩展到未经对照的其他模型。

## 对 Wiki 的映射

- [LingBot-Touch 主实体](../../wiki/entities/paper-lingbot-touch.md)
- [官方项目页](../sites/lingbot-touch-official.md)
- [推理源码与权重边界](../repos/lingbot-touch-official.md)
