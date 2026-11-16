# WAM-TTT：Steering World-Action Models by Watching Human Play at Test Time

> 来源归档（ingest）

- **标题：** WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time
- **类型：** paper
- **arXiv abs：** <https://arxiv.org/abs/2607.06988>
- **arXiv HTML：** <https://arxiv.org/html/2607.06988v2>
- **作者（论文首页）：** Yusen Feng、Bingchen Han、Jiangran Lyu（共同一作）等；通讯 Libin Liu、Yizhou Wang、Zhizheng Zhang、He Wang
- **机构：** 北京大学（PKU）· 银河通用（Galbot）· 中科院自动化所（CASIA）· 清华大学（Tsinghua）
- **骨干：** **LDA-1B**（latent dynamics action WAM，arXiv:2602.12215）
- **入库日期：** 2026-07-16
- **归档更新：** 2026-11-12（通读 v2 全文后改写）
- **核对范围：** 下载 arXiv v2 PDF（首页标题与编号 2607.06988v2 [cs.RO] 10 Jul 2026 已核对）并逐段通读正文 §1–§5（含 Table 1–3、图注）与附录 A（线性注意力等价推导与算法）、B（超参与数据集）、C（Orig./New 完整表）、D（各任务进度评分细则）、E.1–E.6（定性图集、无场景人视频的实验室泛化、六轴扰动、数据配比、架构消融、伪动作消融）；图 1–5 与附录图仅读图注，未逐帧审视；参考文献列表仅扫读。未看补充视频；全文未出现代码或项目页地址，故无代码可核对。
- **通读：** 已完成
- **一句话说明：** 冻结 **LDA WAM** 上外挂 **Spatial-TTT 式 fast-weight 记忆分支**：meta-training 用 **2286 对** 人–机相位同步示教 + **KV 记忆重建损失** 对齐人视频与机器人 Query；部署时仅用 **无标注 egocentric 人视频** 做 **自监督视频预测 TTT** 更新 fast weights，即可 **steer** 新任务变体，无需机器人动作、人手姿态或全模型微调；**9 项真机**（G1 人形 + Galbot 双臂）**New 家庭场景** 平均 **46.2%** progress，显著优于 **WAM-ICL（7.1%）** 与同骨干 **LDA（32.5%）**。

## 核心摘录（面向 wiki 编译）

### 1) 问题：RFM 部署后难以用人类示教快速转向

- **链接：** §1 Introduction
- **摘录要点：** 既有 RFM 知识固化在预训练权重与有限条件接口（语言、目标图、短历史）；转向新任务变体、物体交互或用户偏好需额外机器人示教或全模型微调。人视频是自然接口，但既有路线多靠 co-training / 微调，常依赖手姿、3D 运动或重定向轨迹等额外监督；直接以人视频作条件（引 MimicDroid [18]）则要求预训练期就学会，且上下文长度随示教增多快速增长。作者主张：不把人视频当轨迹模仿，而当部署时通过视频预测学到的记忆。三条贡献见 §1。
- **对 wiki 的映射：** [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 问题定义

### 2) 方法

- **链接：** §3.1–3.3；Figure 2；Eq. 1–9
- **摘录要点：**
  - **架构（§3.1）：** LDA 每个 DiT block 含 video expert 与 action expert，经 joint attention 通信；仅在 video expert 加 TTT 残差，z(ℓ+1)=ẑ(ℓ+1)+Δz_TTT，x(ℓ+1)=x̂(ℓ+1)（Eq. 1）；Δz_TTT=θ_O f_W(θ_Q(z))（Eq. 2），慢投影 θ_{K,V,Q,O} 与快权重网络 f_W。
  - **Meta-training（§3.2）：** 人侧 K_h=θ_K(h)、V_h=θ_V(h)，机器人侧 Q_r=θ_Q(z_r)；L_KVM=‖f_W(K_h)−V_h‖²/(B·L_h·d)（Eq. 3）；内环 L_adapt=L_vg^human+λΣ_ℓ L_KVM（Eq. 4），SGD 更新 W（Eq. 5）；外环 L_meta=L_WAM^robot（Eq. 6，视频扩散 + 动作扩散），反传穿过内环，优化 WAM 参数、θ_{K,V,Q,O} 与 W_init；每个训练样本后快权重丢弃并从 W_init 重置。
  - **相位同步：** 机器人 t 取 φ=t/T_r，选配对人视频中相位最近的帧。
  - **Test-time（§3.3）：** 冻结 WAM、慢投影、W_init 与 action expert，对目标域无动作人视频批次 B_h 在视频生成模式下做与训练同形的内环（Eq. 7–8），仅更新视频侧快权重；N 步后 W_N 固定用于 rollout（Eq. 9）。
- **对 wiki 的映射：** [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 两阶段流程图与损失表

### 3) 实验设置与主结果

- **链接：** §4.1–4.2；Table 1；附录 B、C、D
- **摘录要点：**
  - **平台：** Unitree G1（人形，三指灵巧手 dex-3）、Galbot gripper（双臂二指）、Galbot sharpa（双臂灵巧手，每侧 22 自由度共 58 维）；9 项任务各分配给单一具身（附录 B）：G1 做 Table Bussing / Pour Water / Deliver Drink；gripper 做 Transfer Bottle / Stamp Paper；sharpa 做 Swap Place / Pyramid Stacking / Flip Steak / Multi-step Steak。
  - **数据（附录 B）：** 2286 对人–机 episode（G1 600、gripper 544、sharpa 1142）。机器人数据在标准化 cubicle 内遥操作采集；人侧用 GoPro 第一视角、直接在后来作为 New 评测的真实家庭环境里录制，无手姿 / 关节 / 重定向标注；部署时人侧在部署场景再录一次供 TTT。
  - **评测：** Orig.（同 cubicle，光照 / 桌高 / 物体一致）与 New（未见家庭环境，光照 / 桌高 / 物体同时变化）；指标 progress（%），每格 25 次试验；progress 为按预定义子目标加权的部分给分（附录 D：加权和，权重集中在关键里程碑，由末端位姿与已知物体位姿自动评分）。
  - **Table 1（New 平均）：** WAM-TTT 46.2；LDA 32.5；WAM-Cotrain 25.3；EgoScale 15.0（作者自行复现，原模型未开源）；π0.5 14.8；WAM-ICL 7.1。
  - **Table C.1（Orig. 平均）：** WAM-TTT 61.1；LDA 50.2；WAM-ICL 48.4；π0.5 31.8；EgoScale 30.1；WAM-Cotrain 29.8。New/Orig 保持率：TTT 0.76、LDA 0.65、ICL 0.15、Cotrain 0.85。
  - **逐任务（Table 1，New）：** WAM-TTT 在 Table Bussing 100.0、Deliver Drink 66.7、Swap Place 66.7、Pour Water 30.0、Multi-step Steak 43.8 领先；Transfer Bottle 55.6 低于 LDA 56.0；Stamp Paper 8.3 远低于 LDA 33.3；Pyramid Stacking 10.4 低于 WAM-Cotrain 12.0；Flip Steak 34.3 与 Cotrain 34.2 基本持平。
  - **正文表述与表格不一致：** §4.2 写「9 项里 7 项完全领先，Flip Steak 以 10.0 打平，唯一例外是 Stamp Paper」。按 Table 1 数字，Transfer Bottle 与 Pyramid Stacking 也不是第一，Flip Steak 上 TTT 34.3 而非 10.0（10.0 是 LDA 与 WAM-ICL 的值）。
- **对 wiki 的映射：** [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 结果表与平台矩阵

### 4) 消融与泛化保持

- **链接：** §4.3–4.4；Table 2、3；附录 E
- **摘录要点：**
  - **Table 2（New，每格 10 次试验，Table Bussing / Swap Place）：** 完整 100.0 / 88.9；WAM-LoRA 30.0 / 0.0；w/o Meta Training 9.0 / 0.0；w/o Memory Recon. 66.7 / 72.0；w/o TTT 40.0 / 74.1。注意同为 New 的 Table 1 里 Swap Place 为 66.7（25 次），与 Table 2 的 88.9 不同，论文未解释（试验数不同，推测为不同评测批次）。
  - **Table 3（Deliver Drink 扰动）：** 光照 π0.5 28.0 / LDA 54.0 / WAM-ICL 12.0 / WAM-TTT 66.0；空间 0.0 / 28.0 / 20.0 / 56.0。
  - **E.4 数据配比（New，25 次，3 任务平均）：** (r,h)=(100,0) 59.5；(10,190) 51.4；(100,100) 74.1；(200,0) 73.7；(100,200) 73.3。作者读为「人数据可 1 对 1 替代机器人数据，但需要一定机器人 grounding」。同一总预算下纯机器人数据 (200,0) 与 (100,100) 几乎持平——即该设置下配对人数据与等量机器人数据的收益相当，而非额外增益。
  - **E.5 架构（Table Bussing，10 次）：** 解冻预训练 VLM（ours）100.0；DiT only 72.0；VLM 不预训练 80.0；VLM 冻结 54.0。
  - **E.6 伪动作（New，25 次，4 任务）：** 加 MANO 重定向伪动作 + 前向动力学损失后平均由 72.3 降到 28.9。
  - **E.2 / E.3：** 无场景内人视频直接部署于未见实验室场景，及六轴单因素扰动（物体、光照、高度、底盘 / 背景、新印章、目标位置），仅给定性 rollout 图，无定量表。
- **对 wiki 的映射：** [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 消融与核对

### 5) 超参与线性注意力推导

- **链接：** 附录 A、Table B.1
- **摘录要点：** 骨干 LDA = Qwen3-VL-4B-Instruct + DiT-L MMDiT 动作头；16 个 MMDiT block、hidden 1536、32 头；TTT head dim 48、快权重隐层宽 128；内环 SGD 步数 N=1（meta 与 test-time 相同）；内环学习率 meta 0.1、测试 0.01；λ=4×10⁻²；meta-training 100k 步，全局 batch 128，8×H800。附录 A：线性 f_W(x)=Wx 时 L_KVM 的最小二乘解在 K_h⊤K_h≈(L_h/d)I 的各向同性假设下退化为外积记忆，读出等价于无 softmax 线性注意力；论文自己说明这只在线性特例下严格成立，实际部署的是非线性 MLP。
- **对 wiki 的映射：** [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 工程实践

### 6) 局限与展望

- **链接：** §5
- **摘录要点：** (1) meta-training 配对假设人 episode 与机器人 episode 覆盖相同技能相位分布，错位会使内环信号劣化而损失不报警；(2) 部署适配受快权重网络表达力与 meta-training 固定的慢投影限制，任务偏离 meta 配对分布越远适配越弱，边界未实证刻画；(3) 部署接口仅用第一视角 RGB，未利用手姿、接触、3D 线索。
- **对 wiki 的映射：** [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 局限小节

## 当前提炼状态

- [x] arXiv v2 全文（正文 + 附录 A–E）已通读并对齐摘录
- [ ] 项目页 / 代码：论文全文未给出地址，公开后补链
