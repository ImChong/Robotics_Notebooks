# GR00T 2 官方白皮书（2026 年 11 月，原文阅读归档）

- **正式标题：** GR00T 2: Scaling Robot Foundation Models for Generalization, Dexterity, and Adaptation
- **作者/机构：** NVIDIA 等 / 英伟达（NVIDIA）
- **类型：** NVIDIA GEAR 官方技术白皮书（135 页，2026-11）
- **原文：** <https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf>
- **项目：** <https://research.nvidia.com/labs/gear/gr00t-2/>（[官网归档](../sites/gr00t-2-gear.md)）
- **代码：** 项目页写明 `Code Available in December`；2026-11-10 尚无 GR00T 2 官方运行入口，不得用 Isaac-GR00T N1.7 代替
- **入库/复核日期：** 2026-11-10
- **版本：** 当日 NVIDIA GEAR 发布的 `gr00t2_whitepaper.pdf`；以 PDF 自带页码 1–135、正文 §1–§5 与附录 A–K 为引用基准
- **核对范围：** 2026-11-10 直接读取官方 PDF 的完整可提取文本（约 47 万字符，135 页），逐节检查正文 §1 Introduction、§2 Overview、§3 Capabilities（零/少样本、X-Adapt、RL、Agentic、G1）、§4 How We Made GR00T 2（训练、数据、GearArena、采集、仿真、推理、基础设施）、§5 Conclusion and Open Problems，以及附录 **A Fast-Adapt、B X-Adapt、C Data、D GearArena、E UMIverse、F RL Post-Training、G Agentic GR00T 2、H Humanoid Loco-Manipulation、I Accelerating GR00T 2、J Simulations、K Infrastructure**。重点核对表 1–36、图注和各节局限/统计口径；参考文献仅浏览，PDF 中图像及视频未逐帧/逐像素审阅，部分数学公式的 PDF 字体在纯文本抽取时出现乱码，因而不转载不确定的逐字公式。未运行作者实验和训练代码（代码尚未公开）。
- **通读：** 已完成

## 摘要与方法（原文编译，不转存全文）

- **§2 模型：** 单个 2B 参数的 Cosmos-Predict2.5-2B joint video–action DiT；冻结 Wan 2.1 视频 VAE 与 Cosmos-Reason1 语言编码器，本体专属状态/动作 MLP 连接共享 transformer；历史 0.8 s、5 fps、最多四视角、224×320；未来视频与动作分别构成 flow-matching 目标，动作 token attend 包括生成未来视频在内的视觉 token。实际推理视频约 1–2 步去噪，动作 2–4 步。
- **§2 Turbo：** 同规模 2B Transformer，不是更小的参数量；改用 Wan 2.2 视频 VAE，将空间压缩从每维 8× 提到 16×，视觉 token 约四倍减少，换取速度而有任务精度损失。
- **§4.1 三阶段训练：** 单视角视频预训练 → 多视角视频+动作联合训练 → 提升机器人数据权重的联合训练。只在有有效动作标签的样本施加动作损失；人类视频提供大量物理交互的视觉监督。
- **§4.2/C 数据：** 402,730 h Human-1M 过滤后数据（约 1375 万片段、5.15 万采集人、12.3 万场景）；UMI-10K 10,184 h、94.2 万片段、2897 任务；机器人遥操作 23,500 h、116 万片段，含 YAM 夹爪、Sharpa/Wuji 灵巧手、G1 275 h 和 DROID。Flora 三模型生成时序边界、命令式片段描述和奖励。
- **§4.7/K 训练工程：** 主预训练运行 1024 张 GB300 GPU，DeepSpeed ZeRO-2 同步数据并行；LanceDS 快照/QC 视图/本体 normalization，异步预取、解码和 token packing（32,768 token 行占用率 94.5%）；观察到约 12000 windows/s，暴露的数据等待平均 0.37 ms——**数据通路**指标而非总模型推理吞吐。

## 评测、条件与易误读的结果

| 一手位置 | 核查结果 | 限制/解释 |
|---|---|---|
| §3.1 / A.3 / Fig.2,24 | Dexter-50 零样本平均 **40%**；1-shot **71%**；5-shot **95%**，37/50 项 5-shot 达 100% | **50 项事先选为 5-shot 至少 80% 成功的任务**；不可视为随机任务集的 95% 可靠性。LoRA rank 32，只训练 Transformer 线性层适配器，单张 RTX 5090 约 30 min |
| §3.1 / Tab.1a–b | 6 个 YAM 长程任务平均成功率 **71.7%**；7 个 Dexter 任务 **68.6%**；Turbo 分别 **65.0% / 34.3%** | Toy Truck Assembly 的 65% 是**改过的较容易初始场景**，原场景 GR00T 2 **7/20=35%** |
| §4.2/C.1.5 / Fig.10 | 人类视频占比 0/1/10/100% 时真机成功率 **18/27/28/41%**，任务进度 **50/58/61/70%** | 0% 相比 1% 多了共同预训练阶段差异；1%→10% 真机增益落在误差内；每规模仅一次训练 |
| §3.2 / B | 人类 mocap 手套→Sharpa，UMI→YAM，DexUMI→XHAND/Sharpa，单人类视频→仿真 RL→真实灵巧手 | 无**目标任务**的机器人 teleop 不代表完全无目标本体数据；跨外形 UMI 通常仍需无关任务 play 数据、相机几何匹配与动作语义对齐 |
| §B.2 / Tab.4,17 | UMI→YAM 以 41–75 条 UMI 示范加目标本体无关任务数据后，Dumptruck 60%、Board 50%、Conveyor **86%** | Conveyor 86% 是**物件分类成功占比**（50 件），不同于整个 episode 的成功率；无 wrist warp 时两项装配均 0% |
| §B.3 / Tab.5 | DexUMI → XHAND 香蕉酸奶 **60%**、麦片 **80%**、点火 **75%** | 通过力标签输出 joint torque 与在线柔顺调节；**WAM 本身未把触觉当输入**（§5）；与 Sharpa 跨手转移只用位置，无直接复用 XHAND 力标签 |
| §3.3/F.4 / Fig.5 | **4** 项接触任务，自模仿 RL 成功率 **88、98、94、96%**；SFT 40–76% | 图中每条为 50 次真机试验，残差 SAC 与直接 GRPO / 自模仿训练不同，不能混同 |
| §F.5 / Fig.50,52 | DreamDojo 学习世界模型中在线 RL，RoboLab 10 任务平均 **51.32→70.88%**；真机 4 类任务 **42.5→71.7%** | 学习环境不是无偏实机仿真，仍需少量真机 rollouts 构建模型/校验结果 |
| §3.4/G / Fig.7,8 | 玩具乱序摆放 + SFT/DAgger/Agent 三项 98/100、97/100、97/100；视觉指引对短任务 benchmark 平均 **44.2→76.8%**（加入点标记） | 初始场景与 Tab.1 的摆好的场景不同；感知代理有等待/错误和恢复时间成本。视觉点不表达旋转/路径与双臂同时动作 |
| §3.5/H / Tab.25 | G1 Turbo+SONIC：爆米花 **100% (48/48)**、桌面清理 **93.3% (28/30)**、地面拾取 **83.3% (25/30)**、洒漏清洁 **46.7% (14/30)** | Turbo S1 Jetson Thor 约 1 Hz，生成 1.6 s/80 token，SONIC S0 在 Orin NX **50 Hz**，关节 PD **1000 Hz** |
| §H.3 / Tab.26 | OOD 场景仅真实数据 6.7%，GR00T N1.7 加 sim 40%、Turbo 加 sim **63.3%** | 每格 30 次试验；两套含 sim 策略比较 Fisher 双侧 `p≈0.12`，**差异未有统计显著性证据** |
| §I.1 / Tab.27 | Turbo G1 Thor 120W 中位推理 **123.97ms**，RTX5090 **38.40ms**；Dexter 普通 GR00T2 5090 152.78ms、GB300 48.47ms | 这些是各配置的 **model-prediction latency**；不等于 observation-to-applied-action latency，部分设置图像编码已预取 |
| §I.2 / Tab.28 | 320ms 延迟 noisy-prefix RTC 使 RoboDojo 8 任务成功率 **15.0→21.42%**；p95 跨块指令 jerk 减 **87.2%** | 属仿真延迟模型，未必对应真实采样/链路 |
| §J.2 / Fig.18 | 25 项 YAM、14 checkpoints，仿真对真机任务成功率 Pearson **0.92**，进度与 Elo **0.93** | 只验证该评测家族及配置；相关不等于任意任务的 sim2real 保证 |
| §D / Tab.12–14 | GearArena YAM V2 6035 对盲测 match（35% ties），GR00T2 与 π0.5 相比 +236 Elo，与 N1.7 相比 +95 Elo | Elo 只在同一 suite 内比较；置信区间以**任务/场景** bootstrap，不以每次重复视为独立 |
| §J.3 / Tab.30–34 | RoboCasa365 **67.2%**；RoboDojo 3 layout 平均成功率 **35.52%**；MolmoSpaces **45.12%**；G1 Factory **84.7%** | Tab.19 的 RoboDojo **35.2% SFT、75.8% SFT+RL+Agent** 只测 layout bank 0，**不得同 3-bank 表 31 直接混用** |

## 源码与部署可复现边界

- **论文为方法和实验一手证据，不是代码接口规范。** 官方项目页计划 2026-12 公布代码；本次没有可逐行阅读的 GR00T2 运行实现，源码时序图应标 **不适用**，不能从旧 N1.7 仓库猜想。
- §K.5 提供**论文级** client/server 行为描述：相机/语言/状态传给 policy server，动作 chunk 回 client，按控制周期缓冲/提交；异步请求下一块，并丢弃已过期 waypoint。此设计描述**不等于已经有可运行 CLI/API**。
- §5 明确仍开放：要预测多少未来视频、何种记忆、如何注入触觉反馈、任务可靠性样本量和比较设计、代理检验/恢复延迟；实验自报，未独立复现。

## 对 wiki 的映射

- [GR00T 2：跨本体世界动作基础模型](../../wiki/entities/paper-gr00t-2.md)
